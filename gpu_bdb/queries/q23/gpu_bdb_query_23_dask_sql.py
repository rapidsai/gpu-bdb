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

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q23_utils import (
    q23_year,
    q23_month,
    q23_coefficient,
    read_tables
)

from dask.distributed import wait

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c)

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
    inv_dates_result = c.sql(query_1)

    c.create_table('inv_dates', inv_dates_result, persist=False)
    query_2 = """
        SELECT inv_warehouse_sk,
            inv_item_sk,
            d_moy,
            AVG(CAST(inv_quantity_on_hand AS DOUBLE)) AS q_mean,
            stddev_samp(CAST(inv_quantity_on_hand as DOUBLE)) AS q_std
        FROM inv_dates
        GROUP BY inv_warehouse_sk, inv_item_sk, d_moy
    """
    iteration_1 = c.sql(query_2)

    c.create_table('iteration_1', iteration_1, persist=False)
    query_3 = f"""
        SELECT inv_warehouse_sk,
            inv_item_sk,
            d_moy,
            q_std / q_mean AS qty_cov
        FROM iteration_1
        WHERE (q_std / q_mean) >= {q23_coefficient}
    """

    iteration_2 = c.sql(query_3)

    c.create_table('temp_table', iteration_2, persist=False)
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
    result = c.sql(query)
    result = result.persist()
    wait(result)
    c.drop_table("temp_table")
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
