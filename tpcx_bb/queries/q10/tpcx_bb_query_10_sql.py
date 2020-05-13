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


from dask_cuda import LocalCUDACluster
from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    setup_rmm_pool,
    write_result,
)
from xbb_tools.text import create_sentences_from_reviews, create_words_from_sentences

import dask_cudf
import cudf
import rmm
import cupy as cp
import nvstrings
import os

from dask.distributed import Client, wait

cli_args = tpcxbb_argparser()
# -------- Q10 -----------
eol_char = "è"


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir):

    bc.create_table("product_reviews", data_dir + "product_reviews/*.parquet")

    query = "select pr_item_sk, pr_review_content, pr_review_sk from product_reviews where pr_review_content IS NOT NULL"

    product_reviews_df = bc.sql(query)
    product_reviews_df = bc.partition(
        product_reviews_df, by=["pr_item_sk", "pr_review_content", "pr_review_sk"]
    )

    product_reviews_df[
        "pr_review_content"
    ] = product_reviews_df.pr_review_content.str.lower()
    product_reviews_df[
        "pr_review_content"
    ] = product_reviews_df.pr_review_content.str.replace_multi(
        [".", "?", "!"], eol_char, regex=False
    )

    sentences = product_reviews_df.map_partitions(create_sentences_from_reviews)

    product_reviews_df = product_reviews_df[["pr_item_sk", "pr_review_sk"]]
    product_reviews_df["pr_review_sk"] = product_reviews_df["pr_review_sk"].astype(
        "int32"
    )
    bc.create_table("product_reviews_df", product_reviews_df)

    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]
    bc.create_table("sentences", sentences.compute())

    word_df = sentences.map_partitions(
        create_words_from_sentences,
        global_position_column="sentence_tokenized_global_pos",
    )
    bc.create_table("word_df", word_df)
    word_df = word_df.persist()

    # EDIT THIS PATH ACCORDINGLY, BLAZING EXPECTS ABSOLUTE FILEPATHS
    resources_dir = os.getcwd()
    bc.create_table(
        "negative_sentiment", resources_dir + "/negativeSentiment.txt", names="word"
    )
    bc.create_table(
        "positive_sentiment", resources_dir + "/positiveSentiment.txt", names="word"
    )

    query = """
    SELECT pr_item_sk as item_sk, sentence as review_sentence, sentiment, word as sentiment_word FROM
    (
        SELECT review_idx_global_pos, word, sentiment, sentence FROM
        (
            WITH sent_df AS ( 
                (SELECT word, 'POS' as sentiment FROM positive_sentiment GROUP BY word) 
                UNION ALL
                (SELECT word, 'NEG' as sentiment FROM negative_sentiment GROUP BY word)
            )
            SELECT * FROM word_df INNER JOIN sent_df ON word_df.word = sent_df.word
        ) word_sentence_sentiment 
        LEFT JOIN sentences ON word_sentence_sentiment.sentence_idx_global_pos = sentences.sentence_tokenized_global_pos
    ) temp INNER JOIN product_reviews_df ON temp.review_idx_global_pos = product_reviews_df.pr_review_sk
    ORDER BY item_sk, review_sentence, sentiment, word
    """

    final_df = bc.sql(query)
    return final_df


if __name__ == "__main__":
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        allocator="existing",
        dask_client=client,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )

    result_df = main(cli_args["data_dir"])
    write_result(
        result_df, output_directory=cli_args["output_dir"],
    )

    if cli_args["verify_results"]:
        result_verified = verify_results(cli_args["verify_dir"])
    cli_args["result_verified"] = result_verified
