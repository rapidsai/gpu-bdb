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

# --store_sales date range
q15_startDate = "2001-09-02"
# --+1year
q15_endDate = "2002-09-02"
q15_store_sk = 10

store_sales_cols = ["ss_sold_date_sk", "ss_net_paid", "ss_store_sk", "ss_item_sk"]
date_cols = ["d_date", "d_date_sk"]
item_cols = ["i_item_sk", "i_category_id"]


def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
    item_df = table_reader.read("item", relevant_cols=item_cols)

    if c:
        c.create_table("store_sales", store_sales_df, persist=False)
        c.create_table("date_dim", date_dim_df, persist=False)
        c.create_table("item", item_df, persist=False)

    return store_sales_df, date_dim_df, item_df

