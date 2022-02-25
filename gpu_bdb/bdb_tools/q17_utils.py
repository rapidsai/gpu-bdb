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

q17_gmt_offset = -5.0
# --store_sales date
q17_year = 2001
q17_month = 12

store_sales_cols = [
    "ss_ext_sales_price",
    "ss_sold_date_sk",
    "ss_store_sk",
    "ss_customer_sk",
    "ss_promo_sk",
    "ss_item_sk",
]
item_cols = ["i_category", "i_item_sk"]
customer_cols = ["c_customer_sk", "c_current_addr_sk"]
store_cols = ["s_gmt_offset", "s_store_sk"]
date_cols = ["d_date_sk", "d_year", "d_moy"]
customer_address_cols = ["ca_address_sk", "ca_gmt_offset"]
promotion_cols = ["p_channel_email", "p_channel_dmail", "p_channel_tv", "p_promo_sk"]

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
        backend=config["backend"],
    )

    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    item_df = table_reader.read("item", relevant_cols=item_cols)
    customer_df = table_reader.read("customer", relevant_cols=customer_cols)
    store_df = table_reader.read("store", relevant_cols=store_cols)
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
    customer_address_df = table_reader.read(
        "customer_address", relevant_cols=customer_address_cols
    )
    promotion_df = table_reader.read("promotion", relevant_cols=promotion_cols)

    if c:
        c.create_table("store_sales", store_sales_df, persist=False)
        c.create_table("item", item_df, persist=False)
        c.create_table("customer", customer_df, persist=False)
        c.create_table("store", store_df, persist=False)
        c.create_table("date_dim", date_dim_df, persist=False)
        c.create_table("customer_address", customer_address_df, persist=False)
        c.create_table("promotion", promotion_df, persist=False)

    return (
        store_sales_df,
        item_df,
        customer_df,
        store_df,
        date_dim_df,
        customer_address_df,
        promotion_df,
    )

