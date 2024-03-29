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
    run_query,
)

from bdb_tools.q09_utils import (
    q09_year,
    q09_part1_ca_country,
    q09_part1_ca_state_IN,
    q09_part1_net_profit_min,
    q09_part1_net_profit_max,
    q09_part1_education_status,
    q09_part1_marital_status,
    q09_part1_sales_price_min,
    q09_part1_sales_price_max,
    q09_part2_ca_country,
    q09_part2_ca_state_IN,
    q09_part2_net_profit_min,
    q09_part2_net_profit_max,
    q09_part2_education_status,
    q09_part2_marital_status,
    q09_part2_sales_price_min,
    q09_part2_sales_price_max,
    q09_part3_ca_country,
    q09_part3_ca_state_IN,
    q09_part3_net_profit_min,
    q09_part3_net_profit_max,
    q09_part3_education_status,
    q09_part3_marital_status,
    q09_part3_sales_price_min,
    q09_part3_sales_price_max,
    read_tables
)

def main(client, config):

    (
        store_sales,
        customer_address,
        customer_demographics,
        date_dim,
        store,
    ) = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

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
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
