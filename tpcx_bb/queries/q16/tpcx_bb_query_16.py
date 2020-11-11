#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import sys


from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
    convert_datestring_to_days,
)
from xbb_tools.merge_util import hash_merge
from xbb_tools.readers import build_reader
from dask.distributed import wait

import numpy as np


### conf
q16_date = "2001-03-16"

websale_cols = [
    "ws_order_number",
    "ws_item_sk",
    "ws_warehouse_sk",
    "ws_sold_date_sk",
    "ws_sales_price",
]
web_returns_cols = ["wr_order_number", "wr_item_sk", "wr_refunded_cash"]
date_cols = ["d_date", "d_date_sk"]
item_cols = ["i_item_sk", "i_item_id"]
warehouse_cols = ["w_warehouse_sk", "w_state"]


# INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
# SELECT w_state, i_item_id,
#  SUM(
#    CASE WHEN (unix_timestamp(d_date,'yyyy-MM-dd') < unix_timestamp('${hiveconf:q16_date}','yyyy-MM-dd'))
#    THEN ws_sales_price - COALESCE(wr_refunded_cash,0)
#    ELSE 0.0 END
#  ) AS sales_before,
#  SUM(
#    CASE WHEN (unix_timestamp(d_date,'yyyy-MM-dd') >= unix_timestamp('${hiveconf:q16_date}','yyyy-MM-dd'))
#    THEN ws_sales_price - COALESCE(wr_refunded_cash,0)
#    ELSE 0.0 END
#  ) AS sales_after


def get_before_after_sales(df, q16_timestamp):
    before_flag = df["d_date"] < q16_timestamp
    after_flag = df["d_date"] >= q16_timestamp

    df["sales_before"] = df["sales"].copy()
    df.loc[~before_flag, "sales_before"] = 0.00

    df["sales_after"] = df["sales"].copy()
    df.loc[~after_flag, "sales_after"] = 0.00
    return df


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    web_sales_df = table_reader.read("web_sales", relevant_cols=websale_cols)
    web_returns_df = table_reader.read("web_returns", relevant_cols=web_returns_cols)
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
    item_df = table_reader.read("item", relevant_cols=item_cols)
    warehouse_df = table_reader.read("warehouse", relevant_cols=warehouse_cols)
    return web_sales_df, web_returns_df, date_dim_df, item_df, warehouse_df


def main(client, config):
    import cudf

    web_sales_df, web_returns_df, date_dim_df, item_df, warehouse_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    warehouse_df["w_state_code"] = warehouse_df[["w_state"]].categorize()["w_state"]

    item_df["i_item_id_code"] = item_df[["i_item_id"]].categorize()["i_item_id"]

    ## persisting as you need it for length calculation and to prevent duplicate reading
    ## downstream
    warehouse_df = warehouse_df.persist()

    item_df = item_df.persist()
    ## casting down because of dtype incosistieny in cudf/dask due to cat columns
    ### https://github.com/rapidsai/cudf/issues/4093
    wh_df_codes_min_signed_type = cudf.utils.dtypes.min_signed_type(
        len(warehouse_df["w_state_code"].compute().cat.categories)
    )
    warehouse_df["w_state_code"] = warehouse_df["w_state_code"].cat.codes.astype(
        wh_df_codes_min_signed_type
    )
    unique_states = warehouse_df[["w_state_code", "w_state"]].drop_duplicates()

    warehouse_df = warehouse_df[["w_state_code", "w_warehouse_sk"]]

    ## casting down because of dtype incosistieny in cudf/dask due to cat columns
    ### https://github.com/rapidsai/cudf/issues/4093
    item_df_codes_min_signed_type = cudf.utils.dtypes.min_signed_type(
        len(item_df["i_item_id_code"].compute().cat.categories)
    )
    item_df["i_item_id_code"] = item_df["i_item_id_code"].cat.codes.astype(
        item_df_codes_min_signed_type
    )
    unique_items = item_df[["i_item_id_code", "i_item_id"]].drop_duplicates()
    item_df = item_df[["i_item_id_code", "i_item_sk"]]

    # JOIN date_dim d ON a1.ws_sold_date_sk = d.d_date_sk
    # AND unix_timestamp(d.d_date, 'yyyy-MM-dd') >= unix_timestamp('${hiveconf:q16_date}', 'yyyy-MM-dd') - 30*24*60*60 --subtract 30 days in seconds
    # AND unix_timestamp(d.d_date, 'yyyy-MM-dd') <= unix_timestamp('${hiveconf:q16_date}', 'yyyy-MM-dd') + 30*24*60*60 --add 30 days in seconds

    ##todo: remove below
    date_dim_cov_df = date_dim_df.map_partitions(convert_datestring_to_days)
    q16_timestamp = np.datetime64(q16_date, "D").astype(int)
    filtered_date_df = date_dim_cov_df.query(
        f"d_date >={q16_timestamp- 30} and d_date <= {q16_timestamp+30}",
        meta=date_dim_cov_df._meta,
    ).reset_index(drop=True)

    web_sales_df = web_sales_df.merge(
        filtered_date_df,
        left_on=["ws_sold_date_sk"],
        right_on=["d_date_sk"],
        how="inner",
    )

    cols_2_keep = [
        "ws_order_number",
        "ws_item_sk",
        "ws_warehouse_sk",
        "ws_sales_price",
        "d_date",
    ]

    web_sales_df = web_sales_df[cols_2_keep]
    web_sales_df = web_sales_df.persist()
    wait(web_sales_df)

    # SELECT *
    # FROM web_sales ws
    # LEFT OUTER JOIN web_returns wr ON (ws.ws_order_number = wr.wr_order_number
    # AND ws.ws_item_sk = wr.wr_item_sk)
    # ) a1

    web_sales_web_returns_join = hash_merge(
        lhs=web_sales_df,
        rhs=web_returns_df,
        left_on=["ws_order_number", "ws_item_sk"],
        right_on=["wr_order_number", "wr_item_sk"],
        how="left",
    )
    cols_2_keep = [
        "ws_item_sk",
        "ws_warehouse_sk",
        "ws_sales_price",
        "wr_refunded_cash",
        "d_date",
    ]

    web_sales_web_returns_join = web_sales_web_returns_join[cols_2_keep]
    web_sales_web_returns_join = web_sales_web_returns_join.persist()

    wait(web_sales_web_returns_join)
    del web_sales_df

    # JOIN item i ON a1.ws_item_sk = i.i_item_sk
    web_sales_web_returns_item_join = web_sales_web_returns_join.merge(
        item_df, left_on=["ws_item_sk"], right_on=["i_item_sk"], how="inner"
    )

    cols_2_keep = [
        "ws_warehouse_sk",
        "ws_sales_price",
        "wr_refunded_cash",
        "i_item_id_code",
        "d_date",
    ]

    web_sales_web_returns_item_join = web_sales_web_returns_item_join[cols_2_keep]

    # JOIN warehouse w ON a1.ws_warehouse_sk = w.w_warehouse_sk
    web_sales_web_returns_item_warehouse_join = web_sales_web_returns_item_join.merge(
        warehouse_df,
        left_on=["ws_warehouse_sk"],
        right_on=["w_warehouse_sk"],
        how="inner",
    )

    merged_df = web_sales_web_returns_item_warehouse_join[
        [
            "ws_sales_price",
            "wr_refunded_cash",
            "i_item_id_code",
            "w_state_code",
            "d_date",
        ]
    ]

    merged_df["sales"] = web_sales_web_returns_item_warehouse_join[
        "ws_sales_price"
    ].fillna(0) - web_sales_web_returns_item_warehouse_join["wr_refunded_cash"].fillna(
        0
    )
    sales_df = merged_df[["i_item_id_code", "w_state_code", "d_date", "sales"]]

    sales_before_after_df = sales_df.map_partitions(
        get_before_after_sales, q16_timestamp
    )
    cols_2_keep = ["i_item_id_code", "w_state_code", "sales_before", "sales_after"]
    sales_before_after_df = sales_before_after_df[cols_2_keep]

    ## group by logic
    group_cols = ["w_state_code", "i_item_id_code"]

    agg_df = sales_before_after_df.groupby(group_cols, sort=True).agg(
        {"sales_before": "sum", "sales_after": "sum"}
    )
    agg_df = agg_df.reset_index(drop=False)

    agg_df = agg_df.loc[:99].persist()

    agg_df = agg_df.reset_index(drop=False)
    agg_df.columns = [
        "sorted_grp_index",
        "w_state_code",
        "i_item_id_code",
        "sales_before",
        "sales_after",
    ]
    agg_df = agg_df.merge(unique_states, how="left", on="w_state_code")[
        ["sorted_grp_index", "w_state", "i_item_id_code", "sales_before", "sales_after"]
    ]
    agg_df = agg_df.merge(unique_items, how="left", on="i_item_id_code")[
        ["sorted_grp_index", "w_state", "i_item_id", "sales_before", "sales_after"]
    ]
    agg_df = agg_df.sort_values(by=["sorted_grp_index"])
    ## only 100 rows so computing is fine
    return agg_df[["w_state", "i_item_id", "sales_before", "sales_after"]].compute()


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
