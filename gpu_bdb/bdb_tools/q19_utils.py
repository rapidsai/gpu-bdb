
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

q19_returns_dates_IN = ["2004-03-08", "2004-08-02", "2004-11-15", "2004-12-20"]

eol_char = "è"


def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"], basepath=config["data_dir"], backend=config["backend"],
    )
    date_dim_cols = ["d_week_seq", "d_date_sk", "d_date"]
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_dim_cols)
    store_returns_cols = ["sr_returned_date_sk", "sr_item_sk", "sr_return_quantity"]
    store_returns_df = table_reader.read(
        "store_returns", relevant_cols=store_returns_cols
    )
    web_returns_cols = ["wr_returned_date_sk", "wr_item_sk", "wr_return_quantity"]
    web_returns_df = table_reader.read("web_returns", relevant_cols=web_returns_cols)

    ### splitting by row groups for better parallelism
    pr_table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=True,
        backend=config["backend"],
    )

    product_reviews_cols = ["pr_item_sk", "pr_review_content", "pr_review_sk"]
    product_reviews_df = pr_table_reader.read(
        "product_reviews", relevant_cols=product_reviews_cols
    )

    if c:
        c.create_table('web_returns', web_returns_df, persist=False)
        c.create_table('date_dim', date_dim_df, persist=False)
        c.create_table('product_reviews', product_reviews_df, persist=False)
        c.create_table('store_returns', store_returns_df, persist=False)

    return date_dim_df, store_returns_df, web_returns_df, product_reviews_df

