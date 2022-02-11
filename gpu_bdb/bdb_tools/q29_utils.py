
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

q29_limit = 100


def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"], basepath=config["data_dir"],backend=config["backend"],
    )
    item_cols = ["i_item_sk", "i_category_id"]
    item_df = table_reader.read("item", relevant_cols=item_cols)

    ws_cols = ["ws_order_number", "ws_item_sk"]
    ws_df = table_reader.read("web_sales", relevant_cols=ws_cols)

    if c:
        c.create_table('item', item_df, persist=False)
        c.create_table('web_sales', ws_df, persist=False)

    return item_df, ws_df

