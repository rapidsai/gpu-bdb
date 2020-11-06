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


import rmm
import cupy as cp
import distributed

from xbb_tools.readers import build_reader
from dask.distributed import Client, wait


# -------- Q10 -----------
eol_char = "è"


def read_tables(config):

    ### splitting by row groups for better parallelism
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=True,
    )
    product_reviews_cols = ["pr_item_sk", "pr_review_content", "pr_review_sk"]

    product_reviews_df = table_reader.read(
        "product_reviews", relevant_cols=product_reviews_cols,
    )
    return product_reviews_df


def load_sentiment_words(filename, sentiment):
    import cudf

    with open(filename) as fh:
        sentiment_words = list(map(str.strip, fh.readlines()))
        # dedupe for one extra record in the source file
        sentiment_words = list(set(sentiment_words))

    sent_df = cudf.DataFrame({"word": sentiment_words})
    sent_df["sentiment"] = sentiment
    return sent_df


def main(client, config):
    import cudf
    import dask_cudf

    product_reviews_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    product_reviews_df = product_reviews_df[
        ~product_reviews_df.pr_review_content.isnull()
    ].reset_index(drop=True)

    product_reviews_df[
        "pr_review_content"
    ] = product_reviews_df.pr_review_content.str.lower()
    product_reviews_df[
        "pr_review_content"
    ] = product_reviews_df.pr_review_content.str.replace(
        [".", "?", "!"], [eol_char], regex=False
    )

    sentences = product_reviews_df.map_partitions(create_sentences_from_reviews)
    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]

    word_df = sentences.map_partitions(
        create_words_from_sentences,
        global_position_column="sentence_tokenized_global_pos",
    )

    # These files come from the official TPCx-BB kit
    # We extracted them from bigbenchqueriesmr.jar
    sentiment_dir = os.path.join(config["data_dir"], "sentiment_files")
    neg_sent_df = load_sentiment_words(os.path.join(sentiment_dir, "/negativeSentiment.txt"), "NEG")
    pos_sent_df = load_sentiment_words(os.path.join(sentiment_dir, "/positiveSentiment.txt"), "POS")

    sent_df = cudf.concat([pos_sent_df, neg_sent_df])
    sent_df = dask_cudf.from_cudf(sent_df, npartitions=1)

    word_sentence_sentiment = word_df.merge(sent_df, how="inner", on="word")

    temp = word_sentence_sentiment.merge(
        sentences,
        how="left",
        left_on="sentence_idx_global_pos",
        right_on="sentence_tokenized_global_pos",
    )

    temp = temp[["review_idx_global_pos", "word", "sentiment", "sentence"]]
    product_reviews_df = product_reviews_df[["pr_item_sk", "pr_review_sk"]]
    product_reviews_df["pr_review_sk"] = product_reviews_df["pr_review_sk"].astype(
        "int32"
    )

    final = temp.merge(
        product_reviews_df,
        how="inner",
        left_on="review_idx_global_pos",
        right_on="pr_review_sk",
    )

    final = final.rename(
        columns={
            "pr_item_sk": "item_sk",
            "sentence": "review_sentence",
            "word": "sentiment_word",
        }
    )
    keepcols = ["item_sk", "review_sentence", "sentiment", "sentiment_word"]
    final = final[keepcols].persist()
    # with sf100, there are 3.2M postive and negative review sentences(rows)
    final = final.sort_values(by=keepcols)
    final = final.persist()
    wait(final)
    return final


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
