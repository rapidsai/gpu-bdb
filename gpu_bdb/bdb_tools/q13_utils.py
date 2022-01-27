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

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    date_cols = ["d_date_sk", "d_year"]
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)

    customer_cols = ["c_customer_sk", "c_customer_id", "c_first_name", "c_last_name"]
    customer_df = table_reader.read("customer", relevant_cols=customer_cols)

    s_sales_cols = ["ss_sold_date_sk", "ss_customer_sk", "ss_net_paid"]
    s_sales_df = table_reader.read("store_sales", relevant_cols=s_sales_cols)

    w_sales_cols = ["ws_sold_date_sk", "ws_bill_customer_sk", "ws_net_paid"]
    web_sales_df = table_reader.read("web_sales", relevant_cols=w_sales_cols)

    if c:
        c.create_table("date_dim", date_dim_df, persist=False)
        c.create_table("customer", customer_df, persist=False)
        c.create_table("store_sales", s_sales_df, persist=False)
        c.create_table("web_sales", web_sales_df, persist=False)

    return (date_dim_df, customer_df, s_sales_df, web_sales_df)

