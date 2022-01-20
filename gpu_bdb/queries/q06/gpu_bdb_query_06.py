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

import sys


from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)
from distributed import wait

from bdb_tools.q06_utils import (
    q06_YEAR,
    q06_LIMIT,
    read_tables
)

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


def main(client, config):

    ws_df, ss_df, date_df, customer_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

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

    return result_df.head(q06_LIMIT)


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
