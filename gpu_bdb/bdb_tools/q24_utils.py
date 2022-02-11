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

ws_cols = ["ws_item_sk", "ws_sold_date_sk", "ws_quantity"]
item_cols = ["i_item_sk", "i_current_price"]
imp_cols = [
    "imp_item_sk",
    "imp_competitor_price",
    "imp_start_date",
    "imp_end_date",
    "imp_sk",
]
ss_cols = ["ss_item_sk", "ss_sold_date_sk", "ss_quantity"]

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
        backend=config["backend"],
    )
    ### read tables
    ws_df = table_reader.read("web_sales", relevant_cols=ws_cols)
    item_df = table_reader.read("item", relevant_cols=item_cols)
    imp_df = table_reader.read("item_marketprices", relevant_cols=imp_cols)
    ss_df = table_reader.read("store_sales", relevant_cols=ss_cols)

    if c:
        c.create_table("web_sales", ws_df, persist=False)
        c.create_table("item", item_df, persist=False)
        c.create_table("item_marketprices", imp_df, persist=False)
        c.create_table("store_sales", ss_df, persist=False)

    return ws_df, item_df, imp_df, ss_df

