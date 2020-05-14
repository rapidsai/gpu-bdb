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

import sys
from collections import OrderedDict


from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    left_semi_join,
    run_dask_cudf_query,
)

from xbb_tools.readers import build_reader
from xbb_tools.text import (
    create_sentences_from_reviews,
    create_words_from_sentences,
)
import numpy as np
import cupy as cp
from distributed import wait

cli_args = tpcxbb_argparser()


# -------- Q18 -----------
# -- store_sales date range
q18_startDate = "2001-05-02"
# --+90days
q18_endDate = "2001-09-02"
TEMP_TABLE1 = "TEMP_TABLE1"
EOL_CHAR = "è"


@benchmark(dask_profile=cli_args["dask_profile"])
def read_tables():
    table_reader = build_reader(basepath=cli_args["data_dir"])

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
        basepath=cli_args["data_dir"], split_row_groups=True,
    )

    product_reviews_cols = ["pr_review_date", "pr_review_content", "pr_review_sk"]
    product_reviews = pr_table_reader.read(
        "product_reviews", relevant_cols=product_reviews_cols,
    )

    return store_sales, date_dim, store, product_reviews


def create_found_reshaped_with_global_pos(found, targets):
    """Given the dataframe created by mapping find_targets_in_reviews,
    create a new dataframe in which the nonzero values in each row are exploded
    to get their own row. Each row will contain the word, its mapping in the column order,
    and the pr_review_sk for the review from which it came.
    
    Having these as two separate functions makes managing dask metadata easier.
    """

    target_df = cudf.DataFrame({"word": targets}).reset_index(drop=False)
    target_df.columns = ["word_mapping", "word"]

    df_clean = found.drop(["pr_review_sk"], axis=1)

    row_idxs, col_idxs = df_clean.values.nonzero()

    found_reshaped = cudf.DataFrame(
        {"word_mapping": col_idxs, "pr_review_sk": found["pr_review_sk"].iloc[row_idxs]}
    )
    found_reshaped = found_reshaped.merge(target_df, on="word_mapping", how="inner")[
        ["word", "pr_review_sk"]
    ]
    return found_reshaped


def find_targets_in_reviews_helper(ddf, targets_host, str_col_name="pr_review_content"):
    """returns a N x K matrix, where N is the number of rows in ddf that
    contain one of the target words and K is the number of words in targets.
    
    If a target word is found in a review, the value in that row, column
    is non-zero.
    
    At the end, any row with non-zero values is returned.
    
    """
    from cudf._lib.strings import find_multiple

    lowered = ddf[str_col_name].str.lower()
    targets = cudf.Series(targets_host)

    ## TODO: Do the replace/any in cupy land before going to cuDF
    resdf = cudf.DataFrame.from_gpu_matrix(
        cp.asarray(
            find_multiple.find_multiple(lowered._column, targets._column)
        ).reshape(-1, len(targets))
    )

    resdf = resdf.replace([0, -1], [1, 0])
    found_mask = resdf.any(axis=1)
    resdf["pr_review_sk"] = ddf["pr_review_sk"]
    found = resdf.loc[found_mask]
    return create_found_reshaped_with_global_pos(found, targets_host)


def find_relevant_reviews(df, targets_host, str_col_name="pr_review_content"):
    """
     This function finds the  reviews containg target stores and returns the 
     relevant reviews
    """
    targets = cudf.Series(targets_host)
    targets_lower_cpu = targets.str.lower().tolist()
    reviews_found = find_targets_in_reviews_helper(df, targets_lower_cpu)[
        ["word", "pr_review_sk"]
    ]

    combined = reviews_found.merge(
        df[["pr_review_date", "pr_review_sk"]], how="inner", on=["pr_review_sk"]
    )

    return combined


@benchmark(dask_profile=cli_args["dask_profile"])
def main(client):
    store_sales, date_dim, store, product_reviews = read_tables()
    ### adding a wait call slows this down by 3-4 seconds, removing it for now
    ### Make TEMP_TABLE1

    # filter date table
    q18_startDate_int = np.datetime64(q18_startDate, "ms").astype(int)
    q18_endDate_int = np.datetime64(q18_endDate, "ms").astype(int)

    date_dim_filtered = date_dim.loc[
        (date_dim.d_date.astype("datetime64[ms]").astype("int") >= q18_startDate_int)
        & (date_dim.d_date.astype("datetime64[ms]").astype("int") <= q18_endDate_int)
    ].reset_index(drop=True)

    # build the regression_analysis table
    ss_date_dim_join = left_semi_join(
        store_sales,
        date_dim_filtered,
        left_on=["ss_sold_date_sk"],
        right_on=["d_date_sk"],
    )

    temp = (
        ss_date_dim_join.groupby(["ss_store_sk", "ss_sold_date_sk"],)
        .agg({"ss_net_paid": "sum"})
        .reset_index()
    )

    temp["xx"] = temp.ss_sold_date_sk * temp.ss_sold_date_sk
    temp["xy"] = temp.ss_sold_date_sk * temp.ss_net_paid
    temp.columns = ["ss_store_sk", "x", "y", "xx", "xy"]

    regression_analysis = (
        temp.groupby(["ss_store_sk"])
        .agg({"x": ["count", "sum"], "xy": "sum", "y": "sum", "xx": "sum"})
        .reset_index(drop=False)
    )

    regression_analysis["slope"] = (
        regression_analysis[("x", "count")] * regression_analysis[("xy", "sum")]
        - regression_analysis[("x", "sum")] * regression_analysis[("y", "sum")]
    ) / (
        regression_analysis[("x", "count")] * regression_analysis[("xx", "sum")]
        - regression_analysis[("x", "sum")] * regression_analysis[("x", "sum")]
    )
    regression_analysis = regression_analysis[["ss_store_sk", "slope"]]
    regression_analysis.columns = ["ss_store_sk", "slope"]

    regression_analysis["ss_store_sk"] = regression_analysis["ss_store_sk"].astype(
        "int32"
    )
    store["s_store_sk"] = store["s_store_sk"].astype("int32")
    temp_table1 = store.merge(
        regression_analysis[["ss_store_sk", "slope"]]
        .query("slope <= 0")
        .reset_index(drop=True),
        left_on="s_store_sk",
        right_on="ss_store_sk",
    )
    temp_table1 = temp_table1[["s_store_sk", "s_store_name"]]

    # repartition this table to be one partition, since its only 192 at SF1000
    temp_table1 = temp_table1.repartition(npartitions=1)
    temp_table1 = temp_table1.persist()
    ### Make TEMP_TABLE2
    stores_with_regression = temp_table1
    pr = product_reviews

    # known to be small. very few relevant stores (169) at SF1000
    targets = (
        stores_with_regression.s_store_name.str.lower().unique().compute().tolist()
    )
    n_targets = len(targets)

    no_nulls = pr[~pr.pr_review_content.isnull()].reset_index(drop=True)
    no_nulls["pr_review_sk"] = no_nulls["pr_review_sk"].astype("int32")

    ### perssiting because no_nulls is used twice
    no_nulls = no_nulls.reset_index(drop=True).persist()

    temp_table2_meta_empty_df = cudf.DataFrame(
        {
            "word": ["a"],
            "pr_review_sk": np.ones(1, dtype=np.int64),
            "pr_review_date": ["a"],
        }
    ).head(0)

    ### get relevant reviews
    combined = no_nulls.map_partitions(
        find_relevant_reviews, targets, meta=temp_table2_meta_empty_df,
    )

    stores_with_regression["store_ID"] = stores_with_regression.s_store_sk.astype(
        "str"
    ).str.cat(stores_with_regression.s_store_name, sep="_")
    stores_with_regression[
        "s_store_name"
    ] = stores_with_regression.s_store_name.str.lower()

    # Keep this commented line to illustrate that we could exactly match Spark
    # temp_table2 = temp_table2[['store_ID', 'pr_review_date', 'pr_review_content']]
    temp_table2 = combined.merge(
        stores_with_regression, how="inner", left_on=["word"], right_on=["s_store_name"]
    )

    temp_table2 = temp_table2[["store_ID", "pr_review_date", "pr_review_sk"]]
    temp_table2 = temp_table2.persist()

    ### REAL QUERY (PART THREE)
    no_nulls["pr_review_content"] = no_nulls.pr_review_content.str.replace_multi(
        [". ", "? ", "! "], EOL_CHAR, regex=False
    )
    sentences = no_nulls.map_partitions(create_sentences_from_reviews)

    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]

    with open("negativeSentiment.txt") as fh:
        negativeSentiment = list(map(str.strip, fh.readlines()))
        # dedupe for one extra record in the source file
        negativeSentiment = list(set(negativeSentiment))

    word_df = sentences.map_partitions(
        create_words_from_sentences,
        global_position_column="sentence_tokenized_global_pos",
    )
    sent_df = cudf.DataFrame({"word": negativeSentiment})
    sent_df["sentiment"] = "NEG"
    sent_df = dask_cudf.from_cudf(sent_df, npartitions=1)

    word_sentence_sentiment = word_df.merge(sent_df, how="inner", on="word")

    word_sentence_sentiment["sentence_idx_global_pos"] = word_sentence_sentiment[
        "sentence_idx_global_pos"
    ].astype("int64")
    sentences["sentence_tokenized_global_pos"] = sentences[
        "sentence_tokenized_global_pos"
    ].astype("int64")

    word_sentence_sentiment_with_sentence_info = word_sentence_sentiment.merge(
        sentences,
        how="left",
        left_on="sentence_idx_global_pos",
        right_on="sentence_tokenized_global_pos",
    )
    temp_table2["pr_review_sk"] = temp_table2["pr_review_sk"].astype("int32")

    final = word_sentence_sentiment_with_sentence_info.merge(
        temp_table2[["store_ID", "pr_review_date", "pr_review_sk"]],
        how="inner",
        left_on="review_idx_global_pos",
        right_on="pr_review_sk",
    )

    keepcols = ["store_ID", "pr_review_date", "sentence", "sentiment", "word"]
    final = final[keepcols]
    final.columns = ["s_name", "r_date", "r_sentence", "sentiment", "sentiment_word"]
    final = final.persist()
    wait(final)
    final = final.sort_values(["s_name", "r_date", "r_sentence", "sentiment_word"])
    final = final.persist()
    wait(final)
    print(len(final))
    return final


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client = attach_to_cluster(cli_args)

    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
