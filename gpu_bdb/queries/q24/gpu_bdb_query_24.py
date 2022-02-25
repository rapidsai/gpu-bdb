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
from bdb_tools.q24_utils import read_tables
from distributed import wait

### Current Implimenation Assumption
### Grouped Store sales and web sales of 1 item grouped by `date_sk` should fit in memory as number of dates is limited

## query parameter
q24_i_item_sk = 10000

def get_helper_query_table(imp_df, item_df):
    f_imp_df = (
        imp_df.query(f"imp_item_sk == {q24_i_item_sk}", meta=imp_df._meta)
        .repartition(npartitions=1)
        .persist()
    )

    f_item_df = (
        item_df.query(f"i_item_sk == {q24_i_item_sk}", meta=item_df._meta)
        .repartition(npartitions=1)
        .persist()
    )

    item_imp_join_df = f_item_df.merge(
        f_imp_df, left_on="i_item_sk", right_on="imp_item_sk"
    )

    ### item_imp_join_df only has 4 rows

    item_imp_join_df["price_change"] = (
        item_imp_join_df["imp_competitor_price"] - item_imp_join_df["i_current_price"]
    )
    item_imp_join_df["price_change"] = (
        item_imp_join_df["price_change"] / item_imp_join_df["i_current_price"]
    )

    item_imp_join_df["no_days_comp_price"] = (
        item_imp_join_df["imp_end_date"] - item_imp_join_df["imp_start_date"]
    )

    item_imp_join_df = item_imp_join_df[
        ["i_item_sk", "imp_sk", "imp_start_date", "price_change", "no_days_comp_price"]
    ]
    item_imp_join_df = item_imp_join_df.sort_values(
        by=["i_item_sk", "imp_sk", "imp_start_date"]
    )

    return item_imp_join_df


def get_prev_current_ws(df, websales_col="ws_sum"):
    """
        This function assigns the previous and current web-sales for the merged df
    """

    ### todo: see how toggling assigning scaler vs vector impact performence

    curr_ws_f = (df["ws_sold_date_sk"] >= df["imp_start_date"]) & (
        df["ws_sold_date_sk"] < (df["imp_start_date"] + df["no_days_comp_price"])
    )

    prev_ws_f = (
        df["ws_sold_date_sk"] >= (df["imp_start_date"] - df["no_days_comp_price"])
    ) & (df["ws_sold_date_sk"] < (df["imp_start_date"]))

    df["current_ws_quant"] = 0
    df["current_ws_quant"][curr_ws_f] = df[websales_col][curr_ws_f]

    df["prev_ws_quant"] = 0
    df["prev_ws_quant"][prev_ws_f] = df[websales_col][prev_ws_f]

    return df


def get_prev_current_ss(df, store_sales_col="ss_sum"):
    """
        This function assigns the previous and current store-sales for the merged df
    """
    ### todo: see how toggling assigning scaler vs vector impact performence

    curr_ss_f = (df["ss_sold_date_sk"] >= df["imp_start_date"]) & (
        df["ss_sold_date_sk"] < (df["imp_start_date"] + df["no_days_comp_price"])
    )

    prev_ss_f = (
        df["ss_sold_date_sk"] >= (df["imp_start_date"] - df["no_days_comp_price"])
    ) & (df["ss_sold_date_sk"] < (df["imp_start_date"]))

    df["current_ss_quant"] = 0
    df["current_ss_quant"][curr_ss_f] = df[store_sales_col][curr_ss_f]

    df["prev_ss_quant"] = 0
    df["prev_ss_quant"][prev_ss_f] = df[store_sales_col][prev_ss_f]

    return df


def get_ws(ws_df, item_imp_join_df):
    f_ws_df = ws_df.query(f"ws_item_sk == {q24_i_item_sk}", meta=ws_df._meta)
    ## we know that number of dates is limited and we only have 1 item
    f_ws_g_df = (
        f_ws_df.groupby(["ws_item_sk", "ws_sold_date_sk"])
        .agg({"ws_quantity": "sum"})
        .reset_index(drop=False)
        .repartition(npartitions=1)
        .persist()
    )

    f_ws_g_df = f_ws_g_df.rename(columns={"ws_quantity": "ws_sum"})

    f_ws_item_imp_join_df = f_ws_g_df.merge(
        item_imp_join_df, left_on="ws_item_sk", right_on="i_item_sk", how="inner"
    )
    r_ws = f_ws_item_imp_join_df.map_partitions(get_prev_current_ws)

    r_ws = (
        r_ws.groupby(["ws_item_sk", "imp_sk", "price_change"])
        .agg({"current_ws_quant": "sum", "prev_ws_quant": "sum"})
        .reset_index(drop=False)
    )
    return r_ws


def get_ss(ss_df, item_imp_join_df):
    f_ss_df = ss_df.query(
        f"ss_item_sk == {q24_i_item_sk}", meta=ss_df._meta
    ).reset_index(drop=True)

    f_ss_g_df = (
        f_ss_df.groupby(["ss_item_sk", "ss_sold_date_sk"])
        .agg({"ss_quantity": "sum"})
        .reset_index(drop=False)
        .repartition(npartitions=1)
        .persist()
    )

    ### we know that the the number of dates is limited so below should always fit in memory

    f_ss_g_df = f_ss_g_df.rename(columns={"ss_quantity": "ss_sum"})
    # f_ws_df_grouped_df = f_ws_df.group_by('ws_item_sk')

    f_ss_item_imp_join_df = f_ss_g_df.merge(
        item_imp_join_df, left_on="ss_item_sk", right_on="i_item_sk", how="inner"
    )

    r_ss = f_ss_item_imp_join_df.map_partitions(get_prev_current_ss)

    r_ss = (
        r_ss.groupby(["ss_item_sk", "imp_sk"])
        .agg({"current_ss_quant": "sum", "prev_ss_quant": "sum"})
        .reset_index(drop=False)
    )

    cols_2_keep = ["current_ss_quant", "prev_ss_quant", "ss_item_sk", "imp_sk"]
    r_ss = r_ss[cols_2_keep]

    return r_ss


def main(client, config):

    ws_df, item_df, imp_df, ss_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

    ## helper table
    item_imp_join_df = get_helper_query_table(imp_df, item_df)

    r_ss = get_ss(ss_df, item_imp_join_df)

    r_ws = get_ws(ws_df, item_imp_join_df)

    result_df = r_ws.merge(
        r_ss,
        left_on=["ws_item_sk", "imp_sk"],
        right_on=["ss_item_sk", "imp_sk"],
        how="inner",
        suffixes=("ws", "ss"),
    )

    result_df["cross_price_elasticity"] = (
        result_df["current_ss_quant"]
        + result_df["current_ws_quant"]
        - result_df["prev_ss_quant"]
        - result_df["prev_ws_quant"]
    )
    result_df["cross_price_elasticity"] = result_df["cross_price_elasticity"] / (
        (result_df["prev_ss_quant"] + result_df["prev_ws_quant"])
        * result_df["price_change"]
    )
    final_cols_2_keep = ["ws_item_sk", "cross_price_elasticity"]
    result_df = result_df[final_cols_2_keep]
    result_df = result_df.groupby(["ws_item_sk"]).agg(
        {"cross_price_elasticity": "mean"}
    )
    result_df = result_df.reset_index(drop=False)
    wait(result_df)
    return result_df


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
