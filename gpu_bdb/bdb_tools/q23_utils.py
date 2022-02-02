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

from bdb_tools.readers import build_reader

q23_year = 2001
q23_month = 1
q23_coefficient = 1.3


def read_tables(config, c=None):
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

    if c:
        c.create_table('inventory', inv_df, persist=False)
        c.create_table('date_dim', date_df, persist=False)

    return date_df, inv_df

