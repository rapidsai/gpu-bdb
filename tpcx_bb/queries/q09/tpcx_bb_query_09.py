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

from dask.distributed import Client
import sys

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)
from xbb_tools.readers import build_reader

cli_args = tpcxbb_argparser()


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables():
    table_reader = build_reader(
        data_format=cli_args["file_format"],
        basepath=cli_args["data_dir"],
        split_row_groups=cli_args["split_row_groups"],
    )

    ss_columns = [
        "ss_quantity",
        "ss_sold_date_sk",
        "ss_addr_sk",
        "ss_store_sk",
        "ss_cdemo_sk",
        "ss_sales_price",
        "ss_net_profit",
    ]

    store_sales = table_reader.read("store_sales", relevant_cols=ss_columns)

    ca_columns = ["ca_address_sk", "ca_country", "ca_state"]
    customer_address = table_reader.read("customer_address", relevant_cols=ca_columns)

    cd_columns = ["cd_demo_sk", "cd_marital_status", "cd_education_status"]
    customer_demographics = table_reader.read(
        "customer_demographics", relevant_cols=cd_columns
    )

    dd_columns = ["d_year", "d_date_sk"]
    date_dim = table_reader.read("date_dim", relevant_cols=dd_columns)

    s_columns = ["s_store_sk"]
    store = table_reader.read("store", relevant_cols=s_columns)

    return store_sales, customer_address, customer_demographics, date_dim, store


@benchmark(dask_profile=cli_args["dask_profile"])
def main(client):
    import cudf

    # Conf variables

    q09_year = 2001

    q09_part1_ca_country = "United States"
    q09_part1_ca_state_IN = "KY", "GA", "NM"
    q09_part1_net_profit_min = 0
    q09_part1_net_profit_max = 2000
    q09_part1_education_status = "4 yr Degree"
    q09_part1_marital_status = "M"
    q09_part1_sales_price_min = 100
    q09_part1_sales_price_max = 150

    q09_part2_ca_country = "United States"
    q09_part2_ca_state_IN = "MT", "OR", "IN"
    q09_part2_net_profit_min = 150
    q09_part2_net_profit_max = 3000
    q09_part2_education_status = "4 yr Degree"
    q09_part2_marital_status = "M"
    q09_part2_sales_price_min = 50
    q09_part2_sales_price_max = 200

    q09_part3_ca_country = "United States"
    q09_part3_ca_state_IN = "WI", "MO", "WV"
    q09_part3_net_profit_min = 50
    q09_part3_net_profit_max = 25000
    q09_part3_education_status = "4 yr Degree"
    q09_part3_marital_status = "M"
    q09_part3_sales_price_min = 150
    q09_part3_sales_price_max = 200

    (
        store_sales,
        customer_address,
        customer_demographics,
        date_dim,
        store,
    ) = read_tables()

    date_dim = date_dim.query(
        "d_year==@q09_year", meta=date_dim._meta, local_dict={"q09_year": q09_year}
    ).reset_index(drop=True)
    output_table = store_sales.merge(
        date_dim, left_on=["ss_sold_date_sk"], right_on=["d_date_sk"], how="inner"
    )

    output_table = output_table.drop(
        columns=["d_year", "d_date_sk", "ss_sold_date_sk"]
    )  # Drop the columns that are not needed

    output_table = output_table.merge(
        store, left_on=["ss_store_sk"], right_on=["s_store_sk"], how="inner"
    )

    output_table = output_table.drop(columns=["ss_store_sk", "s_store_sk"])

    output_table = output_table.merge(
        customer_demographics,
        left_on=["ss_cdemo_sk"],
        right_on=["cd_demo_sk"],
        how="inner",
    )

    output_table = output_table[
        (
            (output_table["cd_marital_status"] == q09_part1_marital_status)
            & (output_table["cd_education_status"] == q09_part1_education_status)
            & (output_table["ss_sales_price"] >= q09_part1_sales_price_min)
            & (output_table["ss_sales_price"] <= q09_part1_sales_price_max)
        )
        | (
            (output_table["cd_marital_status"] == q09_part2_marital_status)
            & (output_table["cd_education_status"] == q09_part2_education_status)
            & (output_table["ss_sales_price"] >= q09_part2_sales_price_min)
            & (output_table["ss_sales_price"] <= q09_part2_sales_price_max)
        )
        | (
            (output_table["cd_marital_status"] == q09_part3_marital_status)
            & (output_table["cd_education_status"] == q09_part3_education_status)
            & (output_table["ss_sales_price"] >= q09_part3_sales_price_min)
            & (output_table["ss_sales_price"] <= q09_part3_sales_price_max)
        )
    ].reset_index(drop=True)
    output_table = output_table.drop(
        columns=[
            "ss_cdemo_sk",
            "cd_demo_sk",
            "cd_marital_status",
            "cd_education_status",
            "ss_sales_price",
        ]
    )

    output_table = output_table.merge(
        customer_address,
        left_on=["ss_addr_sk"],
        right_on=["ca_address_sk"],
        how="inner",
    )

    output_table = output_table[
        (
            (output_table["ca_country"] == q09_part1_ca_country)
            & (output_table["ca_state"].str.contains("|".join(q09_part1_ca_state_IN)))
            & (output_table["ss_net_profit"] >= q09_part1_net_profit_min)
            & (output_table["ss_net_profit"] <= q09_part1_net_profit_max)
        )
        | (
            (output_table["ca_country"] == q09_part2_ca_country)
            & (output_table["ca_state"].str.contains("|".join(q09_part2_ca_state_IN)))
            & (output_table["ss_net_profit"] >= q09_part2_net_profit_min)
            & (output_table["ss_net_profit"] <= q09_part2_net_profit_max)
        )
        | (
            (output_table["ca_country"] == q09_part3_ca_country)
            & (output_table["ca_state"].str.contains("|".join(q09_part3_ca_state_IN)))
            & (output_table["ss_net_profit"] >= q09_part3_net_profit_min)
            & (output_table["ss_net_profit"] <= q09_part3_net_profit_max)
        )
    ].reset_index(drop=True)
    output_table = output_table.drop(
        columns=[
            "ss_addr_sk",
            "ca_address_sk",
            "ca_country",
            "ca_state",
            "ss_net_profit",
        ]
    )
    ### this is a scaler so no need to transform
    result = output_table["ss_quantity"].sum().persist()
    result = result.compute()
    result_df = cudf.DataFrame({"sum(ss_quantity)": [result]})

    return result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client, bc = attach_to_cluster(cli_args)

    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
