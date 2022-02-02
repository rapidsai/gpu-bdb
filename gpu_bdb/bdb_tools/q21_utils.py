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

store_sales_cols = [
    "ss_item_sk",
    "ss_store_sk",
    "ss_customer_sk",
    "ss_ticket_number",
    "ss_quantity",
    "ss_sold_date_sk",
]
date_cols = ["d_date_sk", "d_year", "d_moy"]
websale_cols = ["ws_item_sk", "ws_bill_customer_sk", "ws_quantity", "ws_sold_date_sk"]
sr_cols = [
    "sr_item_sk",
    "sr_customer_sk",
    "sr_ticket_number",
    "sr_return_quantity",
    "sr_returned_date_sk",
]
store_cols = ["s_store_name", "s_store_id", "s_store_sk"]
item_cols = ["i_item_id", "i_item_desc", "i_item_sk"]

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
    web_sales_df = table_reader.read("web_sales", relevant_cols=websale_cols)
    store_returns_df = table_reader.read("store_returns", relevant_cols=sr_cols)
    store_table_df = table_reader.read("store", relevant_cols=store_cols)
    item_table_df = table_reader.read("item", relevant_cols=item_cols)

    if c:
        c.create_table("store_sales", store_sales_df, persist=False)
        c.create_table("date_dim", date_dim_df, persist=False)
        c.create_table("item", item_table_df, persist=False)
        c.create_table("web_sales", web_sales_df, persist=False)
        c.create_table("store_returns", store_returns_df, persist=False)
        c.create_table("store", store_table_df, persist=False)

    return (
        store_sales_df,
        date_dim_df,
        web_sales_df,
        store_returns_df,
        store_table_df,
        item_table_df,
    )

