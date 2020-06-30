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
    run_dask_cudf_query,
)
from xbb_tools.readers import build_reader

cli_args = tpcxbb_argparser()

q07_HIGHER_PRICE_RATIO = 1.2
# --store_sales date
q07_YEAR = 2004
q07_MONTH = 7
q07_HAVING_COUNT_GE = 10
q07_LIMIT = 10


@benchmark(dask_profile=cli_args["dask_profile"])
def create_high_price_items_df(item_df):
    grouped_item_df = (
        item_df[["i_category", "i_current_price"]]
        .groupby(["i_category"])
        .agg({"i_current_price": "mean"})
    )
    grouped_item_df = grouped_item_df.rename(columns={"i_current_price": "avg_price"})
    grouped_item_df = grouped_item_df.reset_index(drop=False)

    item_df = item_df.merge(grouped_item_df)
    item_df = item_df[
        item_df["i_current_price"] > item_df["avg_price"] * q07_HIGHER_PRICE_RATIO
    ].reset_index(drop=True)
    high_price_items_df = item_df
    del item_df
    return high_price_items_df


@benchmark(
    dask_profile=cli_args["dask_profile"], compute_result=cli_args["get_read_time"]
)
def read_tables():
    table_reader = build_reader(
        data_format=cli_args["file_format"],
        basepath=cli_args["data_dir"],
        repartition_small_table=cli_args["repartition_small_table"],
        split_row_groups=cli_args["split_row_groups"],
    )

    item_cols = ["i_item_sk", "i_current_price", "i_category"]
    store_sales_cols = ["ss_item_sk", "ss_customer_sk", "ss_sold_date_sk"]
    store_cols = ["s_store_sk"]
    date_cols = ["d_date_sk", "d_year", "d_moy"]
    customer_cols = ["c_customer_sk", "c_current_addr_sk"]
    customer_address_cols = ["ca_address_sk", "ca_state"]

    item_df = table_reader.read("item", relevant_cols=item_cols)
    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    store_df = table_reader.read("store", relevant_cols=store_cols)
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
    customer_df = table_reader.read("customer", relevant_cols=customer_cols)
    customer_address_df = table_reader.read(
        "customer_address", relevant_cols=customer_address_cols
    )

    return (
        item_df,
        store_sales_df,
        store_df,
        date_dim_df,
        customer_df,
        customer_address_df,
    )


@benchmark(dask_profile=cli_args["dask_profile"])
def main(client):
    (
        item_df,
        store_sales_df,
        store_df,
        date_dim_df,
        customer_df,
        customer_address_df,
    ) = read_tables()

    high_price_items_df = create_high_price_items_df(item_df)
    del item_df

    ### Query 0. Date Time Filteration Logic
    filtered_date_df = date_dim_df.query(
        f"d_year == {q07_YEAR} and d_moy == {q07_MONTH}", meta=date_dim_df._meta
    ).reset_index(drop=True)

    ### filtering store sales to above dates
    store_sales_df = store_sales_df.merge(
        filtered_date_df,
        left_on=["ss_sold_date_sk"],
        right_on=["d_date_sk"],
        how="inner",
    )

    ### cols 2 keep after merge
    store_sales_cols = ["ss_item_sk", "ss_customer_sk", "ss_sold_date_sk"]
    store_sales_df = store_sales_df[store_sales_cols]

    #### Query 1. `store_sales` join `highPriceItems`
    store_sales_highPriceItems_join_df = store_sales_df.merge(
        high_price_items_df, left_on=["ss_item_sk"], right_on=["i_item_sk"], how="inner"
    )

    #### Query 2. `Customer` Merge `store_sales_highPriceItems_join_df`
    store_sales_highPriceItems_customer_join_df = store_sales_highPriceItems_join_df.merge(
        customer_df, left_on=["ss_customer_sk"], right_on=["c_customer_sk"], how="inner"
    )

    #### Query 3. `store_sales_highPriceItems_customer_join_df` Merge `Customer Address`
    customer_address_df = customer_address_df[customer_address_df["ca_state"].notnull()]

    final_merged_df = store_sales_highPriceItems_customer_join_df.merge(
        customer_address_df, left_on=["c_current_addr_sk"], right_on=["ca_address_sk"]
    )

    #### Query 4. Final State Grouped Query

    count_df = final_merged_df["ca_state"].value_counts()
    ### number of states is limited=50
    ### so it can remain a cudf frame
    count_df = count_df.compute()
    count_df = count_df[count_df >= q07_HAVING_COUNT_GE]
    count_df = count_df.sort_values(ascending=False)

    result_df = count_df.head(q07_LIMIT)
    result_df = result_df.reset_index(drop=False).rename(
        columns={"index": "ca_state", "ca_state": "cnt"}
    )

    return result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client = attach_to_cluster(cli_args)

    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
