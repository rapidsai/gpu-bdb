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

websale_cols = [
    "ws_order_number",
    "ws_item_sk",
    "ws_warehouse_sk",
    "ws_sold_date_sk",
    "ws_sales_price",
]
web_returns_cols = ["wr_order_number", "wr_item_sk", "wr_refunded_cash"]
date_cols = ["d_date", "d_date_sk"]
item_cols = ["i_item_sk", "i_item_id"]
warehouse_cols = ["w_warehouse_sk", "w_state"]

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    web_sales_df = table_reader.read("web_sales", relevant_cols=websale_cols)
    web_returns_df = table_reader.read("web_returns", relevant_cols=web_returns_cols)
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
    item_df = table_reader.read("item", relevant_cols=item_cols)
    warehouse_df = table_reader.read("warehouse", relevant_cols=warehouse_cols)

    if c:
        c.create_table("web_sales", web_sales_df, persist=False)
        c.create_table("web_returns", web_returns_df, persist=False)
        c.create_table("date_dim", date_dim_df, persist=False)
        c.create_table("item", item_df, persist=False)
        c.create_table("warehouse", warehouse_df, persist=False)

    return web_sales_df, web_returns_df, date_dim_df, item_df, warehouse_df

