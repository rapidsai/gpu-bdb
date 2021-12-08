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

import numpy as np
import sys
import os

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
    convert_datestring_to_days,
)

from bdb_tools.readers import build_reader


# -------- Q22 -----------
q22_date = "2001-05-08"
q22_i_current_price_min = "0.98"
q22_i_current_price_max = "1.5"


def read_tables(data_dir, c, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )
    inv_columns = [
        "inv_item_sk",
        "inv_warehouse_sk",
        "inv_date_sk",
        "inv_quantity_on_hand",
    ]
    inventory = table_reader.read("inventory", relevant_cols=inv_columns)

    item_columns = ["i_item_id", "i_current_price", "i_item_sk"]
    item = table_reader.read("item", relevant_cols=item_columns)

    warehouse_columns = ["w_warehouse_sk", "w_warehouse_name"]
    warehouse = table_reader.read("warehouse", relevant_cols=warehouse_columns)

    dd_columns = ["d_date_sk", "d_date"]
    date_dim = table_reader.read("date_dim", relevant_cols=dd_columns)
    date_dim = date_dim.map_partitions(convert_datestring_to_days)

    c.create_table('inventory', inventory, persist=False)
    c.create_table('item', item, persist=False)
    c.create_table('warehouse', warehouse, persist=False)
    c.create_table('date_dim', date_dim, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

    # Filter limit in days
    min_date = np.datetime64(q22_date, "D").astype(int) - 30
    max_date = np.datetime64(q22_date, "D").astype(int) + 30
    d_date_int = np.datetime64(q22_date, "D").astype(int)
    ratio_min = 2.0 / 3.0
    ratio_max = 3.0 / 2.0
    query = f"""
        SELECT
            w_warehouse_name,
            i_item_id,
            SUM(CASE WHEN d_date - {d_date_int} < 0 THEN inv_quantity_on_hand ELSE 0 END) AS inv_before,
            SUM(CASE WHEN d_date - {d_date_int} >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS inv_after
        FROM
            inventory inv,
            item i,
            warehouse w,
            date_dim d
        WHERE i_current_price BETWEEN {q22_i_current_price_min} AND {q22_i_current_price_max}
        AND i_item_sk        = inv_item_sk
        AND inv_warehouse_sk = w_warehouse_sk
        AND inv_date_sk      = d_date_sk
        AND d_date >= {min_date}
        AND d_date <= {max_date}
        GROUP BY w_warehouse_name, i_item_id
    """
    intermediate = c.sql(query)
    c.create_table("intermediate", intermediate ,persist=False)

    query_2 = f"""
        SELECT
            w_warehouse_name,
            i_item_id,
            inv_before,
            inv_after
        FROM intermediate
        WHERE inv_before > 0
        AND CAST(inv_after AS DOUBLE) / CAST(inv_before AS DOUBLE) >= {ratio_min}
        AND CAST(inv_after AS DOUBLE) / CAST(inv_before AS DOUBLE) <= {ratio_max}
        ORDER BY w_warehouse_name, i_item_id
        LIMIT 100
    """
    result = c.sql(query_2)
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main, sql_context=c)
