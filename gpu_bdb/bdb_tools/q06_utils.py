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

# -------- Q6 -----------
q06_LIMIT = 100
# --web_sales and store_sales date
q06_YEAR = 2001


def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    web_sales_cols = [
        "ws_bill_customer_sk",
        "ws_sold_date_sk",
        "ws_ext_list_price",
        "ws_ext_wholesale_cost",
        "ws_ext_discount_amt",
        "ws_ext_sales_price",
    ]
    store_sales_cols = [
        "ss_customer_sk",
        "ss_sold_date_sk",
        "ss_ext_list_price",
        "ss_ext_wholesale_cost",
        "ss_ext_discount_amt",
        "ss_ext_sales_price",
    ]
    date_cols = ["d_date_sk", "d_year", "d_moy"]
    customer_cols = [
        "c_customer_sk",
        "c_customer_id",
        "c_email_address",
        "c_first_name",
        "c_last_name",
        "c_preferred_cust_flag",
        "c_birth_country",
        "c_login",
    ]

    ws_df = table_reader.read("web_sales", relevant_cols=web_sales_cols)
    ss_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    date_df = table_reader.read("date_dim", relevant_cols=date_cols)
    customer_df = table_reader.read("customer", relevant_cols=customer_cols)

    if c:
        c.create_table('web_sales', ws_df, persist=False)
        c.create_table('store_sales', ss_df, persist=False)
        c.create_table('date_dim', date_df, persist=False)
        c.create_table('customer', customer_df, persist=False)

    return (ws_df, ss_df, date_df, customer_df)

