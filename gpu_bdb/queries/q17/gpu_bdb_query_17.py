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

import cudf

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    left_semi_join,
    run_query,
    )
from bdb_tools.q17_utils import (
    q17_gmt_offset,
    q17_year,
    q17_month,
    store_sales_cols,
    read_tables
)

q17_i_category_IN = "Books", "Music"

def main(client, config):

    (
        store_sales_df,
        item_df,
        customer_df,
        store_df,
        date_dim_df,
        customer_address_df,
        promotion_df,
    ) = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

    # store_sales ss LEFT SEMI JOIN date_dim dd ON ss.ss_sold_date_sk = dd.d_date_sk AND dd.d_year = ${hiveconf:q17_year} AND dd.d_moy = ${hiveconf:q17_month}
    filtered_date_df = date_dim_df.query(
        f"d_year == {q17_year} and d_moy == {q17_month}", meta=date_dim_df._meta
    ).reset_index(drop=True)
    ss_date_join = left_semi_join(
        store_sales_df,
        filtered_date_df,
        left_on="ss_sold_date_sk",
        right_on="d_date_sk",
    )
    ss_date_join = ss_date_join[store_sales_cols]

    # LEFT SEMI JOIN item i ON ss.ss_item_sk = i.i_item_sk AND i.i_category IN (${hiveconf:q17_i_category_IN})
    filtered_item_df = item_df.loc[
        item_df["i_category"].isin(q17_i_category_IN)
    ].reset_index(drop=True)
    ss_date_item_join = left_semi_join(
        ss_date_join, filtered_item_df, left_on="ss_item_sk", right_on="i_item_sk"
    )

    # LEFT SEMI JOIN store s ON ss.ss_store_sk = s.s_store_sk AND s.s_gmt_offset = ${hiveconf:q17_gmt_offset}
    filtered_store_df = store_df.query(
        f"s_gmt_offset == {q17_gmt_offset}", meta=store_df._meta
    ).reset_index(drop=True)

    ss_date_item_store_join = left_semi_join(
        ss_date_item_join,
        filtered_store_df,
        left_on="ss_store_sk",
        right_on="s_store_sk",
    )

    #    (SELECT c.c_customer_sk FROM customer c LEFT SEMI JOIN customer_address ca
    # ON c.c_current_addr_sk = ca.ca_address_sk AND ca.ca_gmt_offset = ${hiveconf:q17_gmt_offset}
    # ) sub_c

    filtered_customer_address = customer_address_df.query(
        f"ca_gmt_offset == {q17_gmt_offset}"
    ).reset_index(drop=True)

    sub_c = left_semi_join(
        customer_df,
        filtered_customer_address,
        left_on="c_current_addr_sk",
        right_on="ca_address_sk",
    )

    # sub_c ON ss.ss_customer_sk = sub_c.c_customer_sk

    ss_date_item_store_customer_join = left_semi_join(
        ss_date_item_store_join,
        sub_c,
        left_on="ss_customer_sk",
        right_on="c_customer_sk",
    )

    # JOIN promotion p ON ss.ss_promo_sk = p.p_promo_sk
    ss_date_item_store_customer_promotion_join = ss_date_item_store_customer_join.merge(
        promotion_df, left_on="ss_promo_sk", right_on="p_promo_sk", how="inner"
    )

    final_df = ss_date_item_store_customer_promotion_join

    # SELECT p_channel_email, p_channel_dmail, p_channel_tv,
    # CASE WHEN (p_channel_dmail = 'Y' OR p_channel_email = 'Y' OR p_channel_tv = 'Y')
    # THEN SUM(ss_ext_sales_price) ELSE 0 END as promotional,
    # SUM(ss_ext_sales_price) total
    # ...
    # GROUP BY p_channel_email, p_channel_dmail, p_channel_tv

    ### filling na because `pandas` and `cudf` ignore nulls when grouping stuff
    final_df["p_channel_email"] = final_df["p_channel_email"].fillna("None")
    final_df["p_channel_dmail"] = final_df["p_channel_dmail"].fillna("None")
    final_df["p_channel_tv"] = final_df["p_channel_tv"].fillna("None")

    # SELECT sum(promotional) as promotional, sum(total) as total,
    #   CASE WHEN sum(total) > 0 THEN 100*sum(promotional)/sum(total)
    #                            ELSE 0.0 END as promo_percent

    group_cols = ["p_channel_email", "p_channel_dmail", "p_channel_tv"]

    ### max group_columnss should be 27 (3*3*3)[N,Y,None]
    ### so computing is fine
    grouped_df = final_df.groupby(by=group_cols).agg(
        {"ss_ext_sales_price": "sum", "ss_ext_sales_price": "sum"}
    )
    grouped_df = grouped_df.compute()

    gr_df = grouped_df.reset_index()
    gr_df = gr_df.rename(columns={"ss_ext_sales_price": "total"})
    prom_flag = (
        (gr_df["p_channel_dmail"] == "Y")
        | (gr_df["p_channel_email"] == "Y")
        | (gr_df["p_channel_tv"] == "Y")
    )

    ### CASE WHEN (p_channel_dmail = 'Y' OR p_channel_email = 'Y' OR p_channel_tv = 'Y')
    ### THEN SUM(ss_ext_sales_price) ELSE 0 END as promotional,
    gr_df["promotional"] = 0
    gr_df["promotional"][prom_flag] = gr_df["total"][prom_flag]

    total_sum = gr_df["total"].sum()
    prom_sum = gr_df["promotional"].sum()
    prom_per = 0
    if prom_sum != 0:
        prom_per = prom_sum / total_sum * 100

    print("Prom SUM = {}".format(prom_sum))
    print("Prom Per = {}".format(prom_per))
    print("Total SUM = {}".format(total_sum))

    return cudf.DataFrame(
        {"promotional": prom_sum, "total": total_sum, "promo_percent": prom_per}
    )


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
