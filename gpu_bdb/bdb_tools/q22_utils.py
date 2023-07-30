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
import pandas as pd

import cudf
import dask_cudf

from bdb_tools.readers import build_reader
from bdb_tools.utils import convert_datestring_to_days

q22_date = "2001-05-08"
q22_i_current_price_min = 0.98
q22_i_current_price_max = 1.5


def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
        backend=config["backend"],
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

    meta_d = {
        "d_date_sk": np.ones(1, dtype=np.int64),
        "d_date": np.ones(1, dtype=np.int64)
    }

    if isinstance(date_dim, dask_cudf.DataFrame):
        meta_df = cudf.DataFrame(meta_d)
    else:
        meta_df = pd.DataFrame(meta_d)

    date_dim = date_dim.map_partitions(convert_datestring_to_days, meta=meta_df)

    if c:
        c.create_table('inventory', inventory, persist=False)
        c.create_table('item', item, persist=False)
        c.create_table('warehouse', warehouse, persist=False)
        c.create_table('date_dim', date_dim, persist=False)

    return inventory, item, warehouse, date_dim

