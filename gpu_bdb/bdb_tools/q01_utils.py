#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# -------- Q1 -----------
q01_i_category_id_IN = 1, 2, 3
# -- sf1 -> 11 stores, 90k sales in 820k lines
q01_ss_store_sk_IN = 10, 20, 33, 40, 50
q01_viewed_together_count = 50
q01_limit = 100


item_cols = ["i_item_sk", "i_category_id"]
ss_cols = ["ss_item_sk", "ss_store_sk", "ss_ticket_number"]


def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_df = table_reader.read("item", relevant_cols=item_cols)
    ss_df = table_reader.read("store_sales", relevant_cols=ss_cols)

    if c:
        c.create_table("item", item_df, persist=False)
        c.create_table("store_sales", ss_df, persist=False)

    return item_df, ss_df

