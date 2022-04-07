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

import cupy as cp
import numpy as np
import cudf
import pandas as pd
from cudf._lib.strings import find_multiple

from bdb_tools.readers import build_reader

q18_startDate = "2001-05-02"
# --+90days
q18_endDate = "2001-09-02"

EOL_CHAR = "è"


def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"], basepath=config["data_dir"], backend=config["backend"],
    )

    store_sales_cols = [
        "ss_store_sk",
        "ss_sold_date_sk",
        "ss_net_paid",
    ]
    date_cols = ["d_date_sk", "d_date"]
    store_cols = ["s_store_sk", "s_store_name"]

    store_sales = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    date_dim = table_reader.read("date_dim", relevant_cols=date_cols)
    store = table_reader.read("store", relevant_cols=store_cols)

    ### splitting by row groups for better parallelism
    pr_table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=True,
        backend=config["backend"],
    )

    product_reviews_cols = ["pr_review_date", "pr_review_content", "pr_review_sk"]
    product_reviews = pr_table_reader.read(
        "product_reviews", relevant_cols=product_reviews_cols,
    )

    if c:
        c.create_table("store", store, persist=False)
        c.create_table("store_sales", store_sales, persist=False)
        c.create_table("date_dim", date_dim, persist=False)
        c.create_table("product_reviews", product_reviews, persist=False)

    return store_sales, date_dim, store, product_reviews


def create_found_reshaped_with_global_pos(found, targets):
    """Given the dataframe created by mapping find_targets_in_reviews,
    create a new dataframe in which the nonzero values in each row are exploded
    to get their own row. Each row will contain the word, its mapping in the column order,
    and the pr_review_sk for the review from which it came.
    Having these as two separate functions makes managing dask metadata easier.
    """
    
    if isinstance(found, cudf.DataFrame):
        target_df = cudf.DataFrame({"word": targets}).reset_index(drop=False)
    else:
        target_df = pd.DataFrame({"word": targets}).reset_index(drop=False)
        
    target_df.columns = ["word_mapping", "word"]

    df_clean = found.drop(["pr_review_sk"], axis=1)

    row_idxs, col_idxs = df_clean.values.nonzero()
    
    if isinstance(found, cudf.DataFrame):
        found_reshaped = cudf.DataFrame(
            {"word_mapping": col_idxs, "pr_review_sk": found["pr_review_sk"].iloc[row_idxs]}
        )
    else:
        found_reshaped = pd.DataFrame(
            {"word_mapping": col_idxs, "pr_review_sk": found["pr_review_sk"].iloc[row_idxs]}
        )
    found_reshaped = found_reshaped.merge(target_df, on="word_mapping", how="inner")[
        ["word", "pr_review_sk"]
    ]
    return found_reshaped

def pandas_find_multiple(lowered, targets):
    tmp = []
    for target in targets:
        tmp.append(lowered.str.find(target))
    
    return [list(x) for x in zip(*tmp)]

def find_targets_in_reviews_helper(ddf, targets, str_col_name="pr_review_content"):
    """returns a N x K matrix, where N is the number of rows in ddf that
    contain one of the target words and K is the number of words in targets.
    
    If a target word is found in a review, the value in that row, column
    is non-zero.
    
    At the end, any row with non-zero values is returned.
    
    """

    lowered = ddf[str_col_name].str.lower()
    ## TODO: Do the replace/any in cupy land before going to cuDF
    
    if isinstance(ddf, cudf.DataFrame):
        resdf = cudf.DataFrame(
            cp.asarray(
                cudf.Series(find_multiple.find_multiple(lowered._column, targets._column)).explode()
            ).reshape(-1, len(targets))
        )
    else:
        resdf = pd.DataFrame(
            np.asarray(
                pd.Series(pandas_find_multiple(lowered, targets)).explode()
            ).reshape(-1, len(targets))
        )

    resdf = resdf.replace([0, -1], [1, 0])
    found_mask = resdf.any(axis=1)
    resdf["pr_review_sk"] = ddf["pr_review_sk"]
    found = resdf.loc[found_mask]
    return create_found_reshaped_with_global_pos(found, targets)


def find_relevant_reviews(df, targets, str_col_name="pr_review_content"):
    """
     This function finds the  reviews containg target stores and returns the 
     relevant reviews
    """
    if isinstance(df, cudf.DataFrame):
        targets = cudf.Series(targets)
    else:
        targets = pd.Series(targets)
    targets_lower = targets.str.lower()
    reviews_found = find_targets_in_reviews_helper(df, targets_lower)[
        ["word", "pr_review_sk"]
    ]

    combined = reviews_found.merge(
        df[["pr_review_date", "pr_review_sk"]], how="inner", on=["pr_review_sk"]
    )

    return combined

