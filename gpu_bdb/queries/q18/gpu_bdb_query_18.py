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

import os

import cudf
import dask_cudf

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    left_semi_join,
    run_query,
)
from bdb_tools.text import (
    create_sentences_from_reviews,
    create_words_from_sentences,
)
from bdb_tools.q18_utils import (
    find_relevant_reviews,
    q18_startDate,
    q18_endDate,
    EOL_CHAR,
    read_tables
)

import numpy as np
from distributed import wait

TEMP_TABLE1 = "TEMP_TABLE1"

def main(client, config):

    store_sales, date_dim, store, product_reviews = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )
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
        stores_with_regression.s_store_name.str.lower()
        .unique()
        .compute()
        .to_arrow()
        .to_pylist()
    )

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
    no_nulls["pr_review_content"] = no_nulls.pr_review_content.str.replace(
        [". ", "? ", "! "], [EOL_CHAR], regex=False
    )
    sentences = no_nulls.map_partitions(create_sentences_from_reviews)

    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]

    # This file comes from the official TPCx-BB kit
    # We extracted it from bigbenchqueriesmr.jar
    sentiment_dir = os.path.join(config["data_dir"], "sentiment_files")
    with open(os.path.join(sentiment_dir, "negativeSentiment.txt")) as fh:
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
    return final


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
