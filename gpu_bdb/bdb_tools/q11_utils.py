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
        backend=config["backend"],
    )

    product_review_cols = [
        "pr_review_rating",
        "pr_item_sk",
    ]
    web_sales_cols = [
        "ws_sold_date_sk",
        "ws_net_paid",
        "ws_item_sk",
    ]
    date_cols = ["d_date_sk", "d_date"]

    pr_df = table_reader.read("product_reviews", relevant_cols=product_review_cols)
    # we only read int columns here so it should scale up to sf-10k as just 26M rows
    pr_df = pr_df.repartition(npartitions=1)

    ws_df = table_reader.read("web_sales", relevant_cols=web_sales_cols)
    date_df = table_reader.read("date_dim", relevant_cols=date_cols)

    if c:
        c.create_table("web_sales", ws_df, persist=False)
        c.create_table("product_reviews", pr_df, persist=False)
        c.create_table("date_dim", date_df, persist=False)

    return (pr_df, ws_df, date_df)
