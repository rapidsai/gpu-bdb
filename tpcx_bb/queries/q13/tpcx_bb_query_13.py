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
from distributed import wait

cli_args = tpcxbb_argparser()

q13_Year = 2001
q13_limit = 100

# Util Function
def get_sales_ratio(df):

    f_year = q13_Year
    s_year = q13_Year + 1

    first_year_flag = df["d_year"] == f_year
    second_year_flag = df["d_year"] == s_year

    df["first_year_sales"] = 0.00
    df["first_year_sales"][first_year_flag] = df["year_total"][first_year_flag]

    df["second_year_sales"] = 0.00
    df["second_year_sales"][second_year_flag] = df["year_total"][second_year_flag]

    return df


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables():
    table_reader = build_reader(
        cli_args["file_format"],
        basepath=cli_args["data_dir"],
        repartition_small_table=cli_args["repartition_small_table"],
        split_row_groups=cli_args["split_row_groups"],
    )

    date_cols = ["d_date_sk", "d_year"]
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)

    customer_cols = ["c_customer_sk", "c_customer_id", "c_first_name", "c_last_name"]
    customer_df = table_reader.read("customer", relevant_cols=customer_cols)

    s_sales_cols = ["ss_sold_date_sk", "ss_customer_sk", "ss_net_paid"]
    s_sales_df = table_reader.read("store_sales", relevant_cols=s_sales_cols)

    w_sales_cols = ["ws_sold_date_sk", "ws_bill_customer_sk", "ws_net_paid"]
    web_sales_df = table_reader.read("web_sales", relevant_cols=w_sales_cols)

    return date_dim_df, customer_df, s_sales_df, web_sales_df


@benchmark(dask_profile=cli_args["dask_profile"])
def main(client):
    date_dim_df, customer_df, s_sales_df, web_sales_df = read_tables()

    # Query 0: time filtration

    filtered_date_df = date_dim_df.query(
        "d_year >= @q13_Year and d_year <= @q13_Year_plus",
        local_dict={"q13_Year": q13_Year, "q13_Year_plus": q13_Year + 1},
        meta=date_dim_df._meta,
    ).reset_index(drop=True)

    s_sales_df = s_sales_df.merge(
        filtered_date_df, how="inner", left_on="ss_sold_date_sk", right_on="d_date_sk"
    )

    web_sales_df = web_sales_df.merge(
        filtered_date_df, how="inner", left_on="ws_sold_date_sk", right_on="d_date_sk"
    )

    # Query 1: Store Sales
    # SELECT
    #   ss.ss_customer_sk AS customer_sk,
    #   sum( case when (d_year = {q13_Year})   THEN ss_net_paid  ELSE 0 END) first_year_total,
    #   sum( case when (d_year = {q13_Year}+1) THEN ss_net_paid  ELSE 0 END) second_year_total
    # FROM store_sales ss
    # JOIN (
    # SELECT d_date_sk, d_year
    # FROM date_dim d
    # WHERE d.d_year in ({q13_Year}, (q13_Year} + 1))) dd on ( ss.ss_sold_date_sk = dd.d_date_sk )
    # GROUP BY ss.ss_customer_sk
    # HAVING first_year_total > 0

    s_grouped_df = (
        s_sales_df.groupby(by=["ss_customer_sk", "d_year"])
        .agg({"ss_net_paid": "sum"})
        .reset_index()
        .rename(columns={"ss_net_paid": "year_total"})
    )

    sales_ratio_df = s_grouped_df.map_partitions(get_sales_ratio)

    sales_ratio_df = (
        sales_ratio_df.groupby(by="ss_customer_sk")
        .agg({"first_year_sales": "max", "second_year_sales": "max"})
        .reset_index()
    )
    sales_ratio_df = sales_ratio_df.query("first_year_sales>0")
    sales_ratio_df["storeSalesIncreaseRatio"] = (
        sales_ratio_df["second_year_sales"] / sales_ratio_df["first_year_sales"]
    )
    sales_ratio_df = sales_ratio_df.drop(
        ["first_year_sales", "second_year_sales"], axis=1
    ).rename(columns={"ss_customer_sk": "c_customer_sk"})

    # Query 2: Web Sales
    # SELECT
    #    ws.ws_bill_customer_sk AS customer_sk,
    #    sum( case when (d_year = {q13_Year})   THEN ws_net_paid  ELSE 0 END) first_year_total,
    #    sum( case when (d_year = {q13_Year}+1) THEN ws_net_paid  ELSE 0 END) second_year_total
    # FROM web_sales ws
    # JOIN (
    # SELECT d_date_sk, d_year
    # FROM date_dim d
    # WHERE d.d_year in ({q13_Year}, ({q13_Year} + 1) )
    # ) dd ON ( ws.ws_sold_date_sk = dd.d_date_sk )
    # GROUP BY ws.ws_bill_customer_sk
    # HAVING first_year_total > 0

    web_grouped_df = (
        web_sales_df.groupby(by=["ws_bill_customer_sk", "d_year"])
        .agg({"ws_net_paid": "sum"})
        .reset_index()
        .rename(columns={"ws_net_paid": "year_total"})
    )

    web_ratio_df = web_grouped_df.map_partitions(get_sales_ratio)

    web_ratio_df = (
        web_ratio_df.groupby(by="ws_bill_customer_sk")
        .agg({"first_year_sales": "max", "second_year_sales": "max"})
        .reset_index()
    )
    web_ratio_df = web_ratio_df.query("first_year_sales>0")
    web_ratio_df["webSalesIncreaseRatio"] = (
        web_ratio_df["second_year_sales"] / web_ratio_df["first_year_sales"]
    )
    web_ratio_df = web_ratio_df.drop(
        ["first_year_sales", "second_year_sales"], axis=1
    ).rename(columns={"ws_bill_customer_sk": "c_customer_sk"})

    # Results Query
    # SELECT
    #   c_customer_sk,
    #   c_first_name,
    #   c_last_name,
    #   (store.second_year_total / store.first_year_total) AS storeSalesIncreaseRatio ,
    #   (web.second_year_total / web.first_year_total) AS webSalesIncreaseRatio
    # FROM store ,
    #      web ,
    #     customer c
    # WHERE store.customer_sk = web.customer_sk
    # AND   web.customer_sk = c_customer_sk
    # AND   (web.second_year_total / web.first_year_total)  >  (store.second_year_total / store.first_year_total)
    # ORDER BY
    # webSalesIncreaseRatio DESC,
    # c_customer_sk,
    # c_first_name,
    # c_last_name
    # LIMIT {q13_limit}

    both_sales = sales_ratio_df.merge(web_ratio_df, how="inner", on="c_customer_sk")

    # need to enforce both being int64 even though both_sales.c_customer_sk is already
    # int64. Figure this out later
    customer_df["c_customer_sk"] = customer_df["c_customer_sk"].astype("int64")
    both_sales["c_customer_sk"] = both_sales["c_customer_sk"].astype("int64")

    final_df = customer_df.merge(both_sales, how="inner", on="c_customer_sk").query(
        "webSalesIncreaseRatio > storeSalesIncreaseRatio"
    )
    final_df = final_df.drop("c_customer_id", axis=1)

    result_df = final_df.repartition(npartitions=1).persist()
    wait(result_df)

    result_df = result_df.map_partitions(
        lambda df: df.sort_values(
            ["webSalesIncreaseRatio", "c_customer_sk", "c_first_name", "c_last_name"],
            ascending=[False, True, True, True],
        )
    )

    result_df = result_df.reset_index(drop=True)
    result_df = result_df.head(q13_limit)
    return result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client = attach_to_cluster(cli_args)

    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
