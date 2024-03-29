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

import numpy as np

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)
from bdb_tools.q22_utils import (
    q22_date,
    q22_i_current_price_min,
    q22_i_current_price_max,
    read_tables
)

def inventory_before_after(df, date):
    df["inv_before"] = df["inv_quantity_on_hand"].copy()
    df.loc[df["d_date"] >= date, "inv_before"] = 0
    df["inv_after"] = df["inv_quantity_on_hand"].copy()
    df.loc[df["d_date"] < date, "inv_after"] = 0
    return df


def main(client, config):
    inventory, item, warehouse, date_dim = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

    item = item.query(
        "i_current_price >= @q22_i_current_price_min and i_current_price<= @q22_i_current_price_max",
        meta=item._meta,
        local_dict={
            "q22_i_current_price_min": q22_i_current_price_min,
            "q22_i_current_price_max": q22_i_current_price_max,
        },
    ).reset_index(drop=True)

    item = item[["i_item_id", "i_item_sk"]]

    output_table = inventory.merge(
        item, left_on=["inv_item_sk"], right_on=["i_item_sk"], how="inner"
    )

    keep_columns = [
        "inv_warehouse_sk",
        "inv_date_sk",
        "inv_quantity_on_hand",
        "i_item_id",
    ]

    output_table = output_table[keep_columns]


    # Filter limit in days
    min_date = np.datetime64(q22_date, "D").astype(int) - 30
    max_date = np.datetime64(q22_date, "D").astype(int) + 30

    date_dim = date_dim.query(
        "d_date>=@min_date and d_date<=@max_date",
        meta=date_dim._meta,
        local_dict={"min_date": min_date, "max_date": max_date},
    ).reset_index(drop=True)

    output_table = output_table.merge(
        date_dim, left_on=["inv_date_sk"], right_on=["d_date_sk"], how="inner"
    )
    keep_columns = ["i_item_id", "inv_quantity_on_hand", "inv_warehouse_sk", "d_date"]
    output_table = output_table[keep_columns]

    output_table = output_table.merge(
        warehouse,
        left_on=["inv_warehouse_sk"],
        right_on=["w_warehouse_sk"],
        how="inner",
    )
    keep_columns = ["i_item_id", "inv_quantity_on_hand", "d_date", "w_warehouse_name"]
    output_table = output_table[keep_columns]

    d_date_int = np.datetime64(q22_date, "D").astype(int)

    output_table = output_table.map_partitions(inventory_before_after, d_date_int)
    keep_columns = ["i_item_id", "w_warehouse_name", "inv_before", "inv_after"]
    output_table = output_table[keep_columns]

    output_table = (
        output_table.groupby(by=["w_warehouse_name", "i_item_id"], sort=True)
        .agg({"inv_before": "sum", "inv_after": "sum"})
        .reset_index()
    )

    output_table["inv_ratio"] = output_table["inv_after"] / output_table["inv_before"]

    ratio_min = 2.0 / 3.0
    ratio_max = 3.0 / 2.0
    output_table = output_table.query(
        "inv_ratio>=@ratio_min and inv_ratio<=@ratio_max",
        meta=output_table._meta,
        local_dict={"ratio_min": ratio_min, "ratio_max": ratio_max},
    )
    keep_columns = [
        "w_warehouse_name",
        "i_item_id",
        "inv_before",
        "inv_after",
    ]
    output_table = output_table[keep_columns]

    ## for query 22 the results vary after 6 th decimal place
    return output_table.head(100)


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
