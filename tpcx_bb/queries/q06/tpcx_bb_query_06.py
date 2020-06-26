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

q06_YEAR = 2001
q6_limit_rows = 100


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

    web_sales_cols = [
        "ws_bill_customer_sk",
        "ws_sold_date_sk",
        "ws_ext_list_price",
        "ws_ext_wholesale_cost",
        "ws_ext_discount_amt",
        "ws_ext_sales_price",
    ]
    store_sales_cols = [
        "ss_customer_sk",
        "ss_sold_date_sk",
        "ss_ext_list_price",
        "ss_ext_wholesale_cost",
        "ss_ext_discount_amt",
        "ss_ext_sales_price",
    ]
    date_cols = ["d_date_sk", "d_year", "d_moy"]
    customer_cols = [
        "c_customer_sk",
        "c_customer_id",
        "c_email_address",
        "c_first_name",
        "c_last_name",
        "c_preferred_cust_flag",
        "c_birth_country",
        "c_login",
    ]

    ws_df = table_reader.read("web_sales", relevant_cols=web_sales_cols)
    ss_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    date_df = table_reader.read("date_dim", relevant_cols=date_cols)
    customer_df = table_reader.read("customer", relevant_cols=customer_cols)

    return (ws_df, ss_df, date_df, customer_df)


def get_sales_ratio(df, table="store_sales"):
    assert table in ("store_sales", "web_sales")

    if table == "store_sales":
        column_prefix = "ss_"
    else:
        column_prefix = "ws_"

    f_year = q06_YEAR
    s_year = q06_YEAR + 1

    first_year_flag = df["d_year"] == f_year
    second_year_flag = df["d_year"] == s_year

    df["first_year_sales"] = 0.00
    df["first_year_sales"][first_year_flag] = (
        (
            df[f"{column_prefix}ext_list_price"][first_year_flag]
            - df[f"{column_prefix}ext_wholesale_cost"][first_year_flag]
            - df[f"{column_prefix}ext_discount_amt"][first_year_flag]
        )
        + df[f"{column_prefix}ext_sales_price"][first_year_flag]
    ) / 2

    df["second_year_sales"] = 0.00
    df["second_year_sales"][second_year_flag] = (
        (
            df[f"{column_prefix}ext_list_price"][second_year_flag]
            - df[f"{column_prefix}ext_wholesale_cost"][second_year_flag]
            - df[f"{column_prefix}ext_discount_amt"][second_year_flag]
        )
        + df[f"{column_prefix}ext_sales_price"][second_year_flag]
    ) / 2

    return df


@benchmark(dask_profile=cli_args["dask_profile"])
def main(client):

    ws_df, ss_df, date_df, customer_df = read_tables()

    filtered_date_df = date_df.query(
        f"d_year >= {q06_YEAR} and d_year <= {q06_YEAR+1}", meta=date_df._meta
    ).reset_index(drop=True)

    web_sales_df = ws_df.merge(
        filtered_date_df, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner"
    )

    ws_grouped_df = (
        web_sales_df.groupby(by=["ws_bill_customer_sk", "d_year"])
        .agg(
            {
                "ws_ext_list_price": "sum",
                "ws_ext_wholesale_cost": "sum",
                "ws_ext_discount_amt": "sum",
                "ws_ext_sales_price": "sum",
            }
        )
        .reset_index()
    )

    web_sales_ratio_df = ws_grouped_df.map_partitions(
        get_sales_ratio, table="web_sales"
    )

    web_sales = (
        web_sales_ratio_df.groupby(["ws_bill_customer_sk"])
        .agg({"first_year_sales": "sum", "second_year_sales": "sum"})
        .reset_index()
    )
    web_sales = web_sales.loc[web_sales["first_year_sales"] > 0].reset_index(drop=True)
    web_sales = web_sales.rename(
        columns={
            "first_year_sales": "first_year_total_web",
            "second_year_sales": "second_year_total_web",
        }
    )

    store_sales_df = ss_df.merge(
        filtered_date_df, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner"
    )

    ss_grouped_df = (
        store_sales_df.groupby(by=["ss_customer_sk", "d_year"])
        .agg(
            {
                "ss_ext_list_price": "sum",
                "ss_ext_wholesale_cost": "sum",
                "ss_ext_discount_amt": "sum",
                "ss_ext_sales_price": "sum",
            }
        )
        .reset_index()
    )

    store_sales_ratio_df = ss_grouped_df.map_partitions(
        get_sales_ratio, table="store_sales"
    )

    store_sales = (
        store_sales_ratio_df.groupby(["ss_customer_sk"])
        .agg({"first_year_sales": "sum", "second_year_sales": "sum"})
        .reset_index()
    )
    store_sales = store_sales.loc[store_sales["first_year_sales"] > 0].reset_index(
        drop=True
    )
    store_sales = store_sales.rename(
        columns={
            "first_year_sales": "first_year_total_store",
            "second_year_sales": "second_year_total_store",
        }
    )

    # SQL "AS"
    sales_df = web_sales.merge(
        store_sales,
        left_on="ws_bill_customer_sk",
        right_on="ss_customer_sk",
        how="inner",
    )
    sales_df["web_sales_increase_ratio"] = (
        sales_df["second_year_total_web"] / sales_df["first_year_total_web"]
    )

    # Join the customer with the combined web and store sales.
    customer_df["c_customer_sk"] = customer_df["c_customer_sk"].astype("int64")
    sales_df["ws_bill_customer_sk"] = sales_df["ws_bill_customer_sk"].astype("int64")
    sales_df = sales_df.merge(
        customer_df,
        left_on="ws_bill_customer_sk",
        right_on="c_customer_sk",
        how="inner",
    ).reset_index(drop=True)

    keep_cols = [
        "ws_bill_customer_sk",
        "web_sales_increase_ratio",
        "c_email_address",
        "c_first_name",
        "c_last_name",
        "c_preferred_cust_flag",
        "c_birth_country",
        "c_login",
    ]

    sales_df = sales_df[keep_cols]
    sales_df = sales_df.rename(columns={"ws_bill_customer_sk": "c_customer_sk"})

    # sales_df is 514,291 rows at SF-100 and 3,031,718 at SF-1000
    # We cant sort descending in Dask right now, anyway
    sales_df = sales_df.repartition(npartitions=1).persist()
    result_df = sales_df.reset_index(drop=True)
    result_df = result_df.map_partitions(
        lambda df: df.sort_values(
            by=[
                "web_sales_increase_ratio",
                "c_customer_sk",
                "c_first_name",
                "c_last_name",
                "c_preferred_cust_flag",
                "c_birth_country",
                "c_login",
            ],
            ascending=False,
        )
    )

    return result_df.head(q6_limit_rows)


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client = attach_to_cluster(cli_args)

    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
