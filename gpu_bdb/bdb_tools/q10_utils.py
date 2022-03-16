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

eol_char = "è"

def read_tables(config, c=None):

    ### splitting by row groups for better parallelism
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=True,
        backend=config["backend"],
    )
    product_reviews_cols = ["pr_item_sk", "pr_review_content", "pr_review_sk"]

    product_reviews_df = table_reader.read(
        "product_reviews", relevant_cols=product_reviews_cols,
    )

    if c:
        c.create_table("product_reviews", product_reviews_df, persist=False)

    return product_reviews_df

