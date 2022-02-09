#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)
from bdb_tools.merge_util import hash_merge

from bdb_tools.q21_utils import read_tables

from dask.distributed import wait

q21_year = 2003
q21_month = 1
q21_limit = 100

def main(client, config):
    (
        store_sales_df,
        date_dim_df,
        web_sales_df,
        store_returns_df,
        store_table_df,
        item_table_df,
    ) = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

    # SELECT sr_item_sk, sr_customer_sk, sr_ticket_number, sr_return_quantity
    # FROM
    # store_returns sr,
    # date_dim d2
    # WHERE d2.d_year = ${hiveconf:q21_year}
    # AND d2.d_moy BETWEEN ${hiveconf:q21_month} AND ${hiveconf:q21_month} + 6 --which were returned in the next six months
    # AND sr.sr_returned_date_sk = d2.d_date_sk
    d2 = date_dim_df.query(
        f"d_year == {q21_year} and d_moy >= {q21_month} and d_moy <= {q21_month+6}",
        meta=date_dim_df._meta,
    ).reset_index(drop=True)

    part_sr = store_returns_df.merge(
        d2, left_on="sr_returned_date_sk", right_on="d_date_sk", how="inner"
    )

    cols_2_keep = [
        "sr_item_sk",
        "sr_customer_sk",
        "sr_ticket_number",
        "sr_return_quantity",
    ]

    part_sr = part_sr[cols_2_keep]

    part_sr = part_sr.persist()
    wait(part_sr)

    # SELECT
    # ws_item_sk, ws_bill_customer_sk, ws_quantity
    # FROM
    # web_sales ws,
    # date_dim d3
    # WHERE d3.d_year BETWEEN ${hiveconf:q21_year} AND ${hiveconf:q21_year} + 2 -- in the following three years (re-purchased by the returning customer afterwards through
    # the web sales channel)
    #   AND ws.ws_sold_date_sk = d3.d_date_sk
    # ) part_ws
    d3 = date_dim_df.query(
        f"d_year >= {q21_year} and d_year <= {q21_year + 2}", meta=date_dim_df._meta
    )
    part_ws = web_sales_df.merge(
        d3, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner"
    )
    cols_2_keep = ["ws_item_sk", "ws_bill_customer_sk", "ws_quantity"]
    part_ws = part_ws[cols_2_keep]
    part_ws = part_ws.persist()
    wait(part_ws)

    # part_ws ON (
    # part_sr.sr_item_sk = part_ws.ws_item_sk
    # AND part_sr.sr_customer_sk = part_ws.ws_bill_customer_sk
    part_ws_part_sr_m = hash_merge(
        lhs=part_sr,
        rhs=part_ws,
        left_on=["sr_item_sk", "sr_customer_sk"],
        right_on=["ws_item_sk", "ws_bill_customer_sk"],
        how="inner",
    )

    cols_2_keep = [
        "sr_item_sk",
        "sr_customer_sk",
        "sr_ticket_number",
        "sr_return_quantity",
        "ws_quantity",
    ]
    part_ws_part_sr_m = part_ws_part_sr_m[cols_2_keep]

    part_ws_part_sr_m = part_ws_part_sr_m.persist()
    wait(part_ws_part_sr_m)
    del part_sr, part_ws
    # SELECT ss_item_sk, ss_store_sk, ss_customer_sk, ss_ticket_number, ss_quantity
    # FROM
    # store_sales ss,
    # date_dim d1
    # WHERE d1.d_year = ${hiveconf:q21_year}
    # AND d1.d_moy = ${hiveconf:q21_month}
    # AND ss.ss_sold_date_sk = d1.d_date_sk
    # ) part_ss
    d1 = date_dim_df.query(
        f"d_year == {q21_year} and d_moy == {q21_month} ", meta=date_dim_df._meta
    )

    part_ss = store_sales_df.merge(
        d1, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner"
    )

    cols_2_keep = [
        "ss_item_sk",
        "ss_store_sk",
        "ss_customer_sk",
        "ss_ticket_number",
        "ss_quantity",
    ]
    part_ss = part_ss[cols_2_keep]

    # part_ss ON (
    # part_ss.ss_ticket_number = part_sr.sr_ticket_number
    # AND part_ss.ss_item_sk = part_sr.sr_item_sk
    # AND part_ss.ss_customer_sk = part_sr.sr_customer_sk

    part_ws_part_sr_m_part_ss_join_df = hash_merge(
        lhs=part_ss,
        rhs=part_ws_part_sr_m,
        left_on=["ss_ticket_number", "ss_item_sk", "ss_customer_sk"],
        right_on=["sr_ticket_number", "sr_item_sk", "sr_customer_sk"],
        how="inner",
    )
    cols_2_keep = [
        "ss_store_sk",
        "ss_quantity",
        "sr_return_quantity",
        "ws_quantity",
        "ss_item_sk",
    ]
    part_ws_part_sr_m_part_ss_join_df = part_ws_part_sr_m_part_ss_join_df[cols_2_keep]

    # INNER JOIN store part_s ON (
    #  part_s.s_store_sk = part_ss.ss_store_sk
    # )
    part_ws_part_sr_m_part_ss_part_s_join_df = store_table_df.merge(
        part_ws_part_sr_m_part_ss_join_df,
        left_on="s_store_sk",
        right_on="ss_store_sk",
        how="inner",
    )

    cols_2_keep = [
        "s_store_name",
        "sr_return_quantity",
        "ss_quantity",
        "ws_quantity",
        "s_store_id",
        "ss_item_sk",
    ]
    part_ws_part_sr_m_part_ss_part_s_join_df = part_ws_part_sr_m_part_ss_part_s_join_df[
        cols_2_keep
    ]

    # INNER JOIN item part_i ON (
    # part_i.i_item_sk = part_ss.ss_item_sk
    # )
    final_df = item_table_df.merge(
        part_ws_part_sr_m_part_ss_part_s_join_df,
        left_on="i_item_sk",
        right_on="ss_item_sk",
        how="inner",
    )
    # GROUP BY
    #  part_i.i_item_id,
    #  part_i.i_item_desc,
    #  part_s.s_store_id,
    #  part_s.s_store_name
    # ORDER BY
    #  part_i.i_item_id,
    #  part_i.i_item_desc,
    #  part_s.s_store_id,
    #  part_s.s_store_name

    cols_2_keep = [
        "i_item_id",
        "i_item_desc",
        "s_store_name",
        "ss_quantity",
        "sr_return_quantity",
        "ws_quantity",
        "s_store_id",
    ]
    grouped_df = final_df[cols_2_keep]
    agg_df = grouped_df.groupby(
        by=["i_item_id", "i_item_desc", "s_store_id", "s_store_name"]
    ).agg({"ss_quantity": "sum", "sr_return_quantity": "sum", "ws_quantity": "sum"})

    agg_df = agg_df.repartition(npartitions=1).persist()

    sorted_agg_df = agg_df.reset_index().map_partitions(
        lambda df: df.sort_values(
            by=["i_item_id", "i_item_desc", "s_store_id", "s_store_name"]
        )
    )

    sorted_agg_df = sorted_agg_df.head(q21_limit)
    sorted_agg_df = sorted_agg_df.rename(
        columns={
            "ss_quantity": "store_sales_quantity",
            "sr_return_quantity": "store_returns_quantity",
            "ws_quantity": "web_sales_quantity",
        }
    )
    sorted_agg_df["i_item_desc"] = sorted_agg_df["i_item_desc"].str.strip()

    return sorted_agg_df


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
