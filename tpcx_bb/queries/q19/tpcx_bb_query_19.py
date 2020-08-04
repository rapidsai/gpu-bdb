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

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)
from xbb_tools.text import create_sentences_from_reviews, create_words_from_sentences


from xbb_tools.readers import build_reader
from dask.distributed import Client, wait
import distributed


# -------- Q19 -----------
q19_returns_dates = ["2004-03-08", "2004-08-02", "2004-11-15", "2004-12-20"]
eol_char = "è"


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"], basepath=config["data_dir"],
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
    )

    product_reviews_cols = ["pr_item_sk", "pr_review_content", "pr_review_sk"]
    product_reviews = pr_table_reader.read(
        "product_reviews", relevant_cols=product_reviews_cols
    )

    return date_dim_df, store_returns_df, web_returns_df, product_reviews


def main(client, config):
    import cudf
    import dask_cudf

    date_dim_df, store_returns_df, web_returns_df, product_reviews_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    # filter date table
    date_dim_df = date_dim_df.merge(
        date_dim_df, on=["d_week_seq"], how="outer", suffixes=("", "_r")
    )
    date_dim_df = date_dim_df[date_dim_df.d_date_r.isin(q19_returns_dates)].reset_index(
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
    sentiment_dir = "/".join(config["data_dir"].split("/")[:-3] + ["sentiment_files"])
    with open(f"{sentiment_dir}/negativeSentiment.txt") as fh:
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
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
