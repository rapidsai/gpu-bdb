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
    run_query,
)
from bdb_tools.text import create_sentences_from_reviews, create_words_from_sentences
from bdb_tools.q19_utils import (
    q19_returns_dates_IN,
    eol_char,
    read_tables
)

from dask.distributed import wait

def main(client, config):

    date_dim_df, store_returns_df, web_returns_df, product_reviews_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

    # filter date table
    date_dim_df = date_dim_df.merge(
        date_dim_df, on=["d_week_seq"], how="outer", suffixes=("", "_r")
    )
    date_dim_df = date_dim_df[date_dim_df.d_date_r.isin(q19_returns_dates_IN)].reset_index(
        drop=True
    )

    date_dim_df = date_dim_df[["d_date_sk"]].drop_duplicates()
    sr_merged_df = store_returns_df.merge(
        date_dim_df,
        left_on=["sr_returned_date_sk"],
        right_on=["d_date_sk"],
        how="inner",
    )
    sr_merged_df = sr_merged_df[["sr_item_sk", "sr_return_quantity"]]
    sr_grouped_df = (
        sr_merged_df.groupby(["sr_item_sk"])
        .agg({"sr_return_quantity": "sum"})
        .reset_index()
        .rename(columns={"sr_return_quantity": "sr_item_qty"})
    )
    sr_grouped_df = sr_grouped_df[sr_grouped_df["sr_item_qty"] > 0]

    wr_merged_df = web_returns_df.merge(
        date_dim_df,
        left_on=["wr_returned_date_sk"],
        right_on=["d_date_sk"],
        how="inner",
    )
    wr_merged_df = wr_merged_df[["wr_item_sk", "wr_return_quantity"]]
    wr_grouped_df = (
        wr_merged_df.groupby(["wr_item_sk"])
        .agg({"wr_return_quantity": "sum"})
        .reset_index()
        .rename(columns={"wr_return_quantity": "wr_item_qty"})
    )
    wr_grouped_df = wr_grouped_df[wr_grouped_df["wr_item_qty"] > 0].reset_index(
        drop=True
    )

    sr_wr_merged_df = sr_grouped_df.merge(
        wr_grouped_df, left_on=["sr_item_sk"], right_on=["wr_item_sk"], how="inner"
    )
    sr_wr_merged_df = sr_wr_merged_df[["sr_item_sk", "sr_item_qty", "wr_item_qty"]]

    product_reviews_df = product_reviews_df[
        ~product_reviews_df.pr_review_content.isnull()
    ].reset_index(drop=True)

    product_reviews_df["pr_item_sk"] = product_reviews_df["pr_item_sk"].astype("int32")
    sr_wr_merged_df["sr_item_sk"] = sr_wr_merged_df["sr_item_sk"].astype("int32")

    merged_df = product_reviews_df.merge(
        sr_wr_merged_df, left_on=["pr_item_sk"], right_on=["sr_item_sk"], how="inner"
    )
    cols_keep = [
        "pr_item_sk",
        "pr_review_content",
        "pr_review_sk",
        "sr_item_qty",
        "wr_item_qty",
    ]
    merged_df = merged_df[cols_keep]
    merged_df["tolerance_flag"] = (
        (merged_df["sr_item_qty"] - merged_df["wr_item_qty"])
        / ((merged_df["sr_item_qty"] + merged_df["wr_item_qty"]) / 2)
    ).abs() <= 0.1
    merged_df = merged_df[merged_df["tolerance_flag"] == True].reset_index(drop=True)
    merged_df = merged_df[["pr_item_sk", "pr_review_content", "pr_review_sk"]]
    merged_df["pr_review_content"] = merged_df.pr_review_content.str.lower()
    merged_df["pr_review_content"] = merged_df.pr_review_content.str.replace(
        [".", "?", "!"], [eol_char], regex=False
    )

    sentences = merged_df.map_partitions(create_sentences_from_reviews)

    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]

    word_df = sentences.map_partitions(
        create_words_from_sentences,
        global_position_column="sentence_tokenized_global_pos",
    )

    # This file comes from the official TPCx-BB kit
    # We extracted it from bigbenchqueriesmr.jar
    sentiment_dir = os.path.join(config["data_dir"], "sentiment_files")
    with open(os.path.join(sentiment_dir, "negativeSentiment.txt")) as fh:
        negativeSentiment = list(map(str.strip, fh.readlines()))
        # dedupe for one extra record in the source file
        negativeSentiment = list(set(negativeSentiment))

    sent_df = cudf.DataFrame({"word": negativeSentiment})
    sent_df["sentiment"] = "NEG"
    sent_df = dask_cudf.from_cudf(sent_df, npartitions=1)

    word_sentence_sentiment = word_df.merge(sent_df, how="inner", on="word")

    merged_df["pr_review_sk"] = merged_df["pr_review_sk"].astype("int32")

    temp = word_sentence_sentiment.merge(
        sentences,
        how="left",
        left_on="sentence_idx_global_pos",
        right_on="sentence_tokenized_global_pos",
    )

    temp = temp[["review_idx_global_pos", "word", "sentiment", "sentence"]]
    merged_df = merged_df[["pr_item_sk", "pr_review_sk"]]

    final = temp.merge(
        merged_df, how="inner", left_on="review_idx_global_pos", right_on="pr_review_sk"
    )
    final = final.rename(
        columns={
            "pr_item_sk": "item_sk",
            "sentence": "review_sentence",
            "word": "sentiment_word",
        }
    )
    keepcols = ["item_sk", "review_sentence", "sentiment", "sentiment_word"]
    final = final[keepcols]
    final = final.persist()
    final = final.sort_values(by=keepcols)
    wait(final)
    return final


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
