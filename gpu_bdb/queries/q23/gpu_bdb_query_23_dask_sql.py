#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Copyright (c) 2019-2020, BlazingSQL, Inc.
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
import os

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader

from dask.distributed import wait


# -------- Q23 -----------
q23_year = 2001
q23_month = 1
q23_coefficient = 1.3


def read_tables(data_dir, bc, config):
    table_reader = build_reader(
        data_format=config["file_format"], basepath=config["data_dir"],
    )

    date_cols = ["d_date_sk", "d_year", "d_moy"]
    date_df = table_reader.read("date_dim", relevant_cols=date_cols)

    inv_cols = [
        "inv_warehouse_sk",
        "inv_item_sk",
        "inv_date_sk",
        "inv_quantity_on_hand",
    ]
    inv_df = table_reader.read("inventory", relevant_cols=inv_cols)

    bc.create_table('inventory', inv_df, persist=False)
    bc.create_table('date_dim', date_df, persist=False)

    # bc.create_table('inventory', os.path.join(data_dir, "inventory/*.parquet"))
    # bc.create_table('date_dim', os.path.join(data_dir, "date_dim/*.parquet"))


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, config, dask_profile=config["dask_profile"])

    query_1 = f"""
        SELECT inv_warehouse_sk,
            inv_item_sk,
            inv_quantity_on_hand,
            d_moy
        FROM inventory inv
        INNER JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk
        AND d.d_year = {q23_year}
        AND d_moy between {q23_month} AND {q23_month + 1}
    """
    inv_dates_result = bc.sql(query_1)

    bc.create_table('inv_dates', inv_dates_result, persist=False)
    query_2 = """
        SELECT inv_warehouse_sk,
            inv_item_sk,
            d_moy,
            AVG(CAST(inv_quantity_on_hand AS DOUBLE)) AS q_mean,
            stddev_samp(CAST(inv_quantity_on_hand as DOUBLE)) AS q_std
        FROM inv_dates
        GROUP BY inv_warehouse_sk, inv_item_sk, d_moy
    """
    iteration_1 = bc.sql(query_2)

    bc.create_table('iteration_1', iteration_1, persist=False)
    query_3 = f"""
        SELECT inv_warehouse_sk,
            inv_item_sk,
            d_moy,
            q_std / q_mean AS qty_cov
        FROM iteration_1
        WHERE (q_std / q_mean) >= {q23_coefficient}
    """

    iteration_2 = bc.sql(query_3)

    bc.create_table('temp_table', iteration_2, persist=False)
    query = f"""
        SELECT inv1.inv_warehouse_sk,
            inv1.inv_item_sk,
            inv1.d_moy,
            inv1.qty_cov AS cov,
            inv2.d_moy AS inv2_d_moy,
            inv2.qty_cov AS inv2_cov
        FROM temp_table inv1
        INNER JOIN temp_table inv2 ON inv1.inv_warehouse_sk = inv2.inv_warehouse_sk
        AND inv1.inv_item_sk = inv2.inv_item_sk
        AND inv1.d_moy = {q23_month}
        AND inv2.d_moy = {q23_month + 1}
        ORDER BY inv1.inv_warehouse_sk,
            inv1.inv_item_sk
    """
    result = bc.sql(query)
    result = result.persist()
    wait(result)
    bc.drop_table("temp_table")
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
