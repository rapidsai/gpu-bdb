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


from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import os
import dask_cudf
from xbb_tools.text import create_sentences_from_reviews, create_words_from_sentences

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    write_result,
)

cli_args = tpcxbb_argparser()


@benchmark(dask_profile=cli_args["dask_profile"])
def read_tables(data_dir):
    bc.create_table("web_returns", data_dir + "web_returns/*.parquet")
    bc.create_table("date_dim", data_dir + "date_dim/*.parquet")
    bc.create_table("product_reviews", data_dir + "product_reviews/*.parquet")
    bc.create_table("store_returns", data_dir + "store_returns/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir):
    read_tables(data_dir)

    query = """
		WITH dateFilter AS 
		(
		    -- within the week ending a given date
			SELECT d1.d_date_sk
			FROM date_dim d1, date_dim d2
			WHERE d1.d_week_seq = d2.d_week_seq
			AND CAST(d2.d_date AS DATE) IN (DATE '2004-03-08', DATE '2004-08-02', DATE '2004-11-15', DATE '2004-12-20')
		), fsr AS
		(
		    --store returns in week ending given date
		    SELECT sr_item_sk, SUM(sr_return_quantity) sr_item_qty
		    FROM store_returns sr
		    INNER JOIN dateFilter d
		    ON sr.sr_returned_date_sk = d.d_date_sk
		    GROUP BY sr_item_sk --across all store and web channels
		    HAVING SUM(sr_return_quantity) > 0
		), fwr AS
		(
		    --web returns in week ending given date
		    SELECT wr_item_sk, SUM(wr_return_quantity) wr_item_qty
		    FROM web_returns wr
		    INNER JOIN dateFilter d
		    ON wr.wr_returned_date_sk = d_date_sk
		    GROUP BY wr_item_sk  --across all store and web channels
		    HAVING SUM(wr_return_quantity) > 0
		), extract_sentiment AS
		(
		    SELECT pr.pr_item_sk, pr.pr_review_content, pr.pr_review_sk
		    FROM product_reviews pr
		    INNER JOIN fsr
		    ON pr.pr_item_sk = fsr.sr_item_sk
		    INNER JOIN fwr
		    ON fsr.sr_item_sk = fwr.wr_item_sk 
		    WHERE pr.pr_review_content IS NOT NULL ---- add as rapids
		    AND abs( CAST((sr_item_qty-wr_item_qty) AS DOUBLE) / ((sr_item_qty + wr_item_qty)/2) ) <= 0.1
		)
		SELECT * FROM extract_sentiment
	"""

    merged_df = bc.sql(query)

    eol_char = "è"
    merged_df["pr_review_sk"] = merged_df["pr_review_sk"].astype("int32")
    merged_df["pr_review_content"] = merged_df.pr_review_content.str.lower()
    merged_df["pr_review_content"] = merged_df.pr_review_content.str.replace_multi(
        [".", "?", "!"], eol_char, regex=False
    )

    sentences = merged_df.map_partitions(create_sentences_from_reviews)
    # need the global position in the sentence tokenized df
    sentences["x"] = 1

    # sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()  # ERROR when partitions are empty
    sentences = sentences.compute()
    sentences["sentence_tokenized_global_pos"] = sentences["x"].cumsum()

    del sentences["x"]
    # 108 is the max number of partitions
    sentences = dask_cudf.from_cudf(sentences, npartitions=108)
    word_df = sentences.map_partitions(
        create_words_from_sentences,
        global_position_column="sentence_tokenized_global_pos",
    )

    resources_dir = os.getcwd()
    bc.create_table("sentences_df", sentences)
    bc.create_table("word_df", word_df)
    bc.create_table(
        "sent_df",
        resources_dir + "/negativeSentiment.txt",
        names=["word"],
        dtype=["str"],
    )
    bc.create_table("merged_df", merged_df)

    last_query = """
	    WITH negativesent AS 
	    (
	        SELECT distinct word 
	        FROM sent_df
	    ), word_sentence_sentiment AS
	    (
	        SELECT sd.word,
	        	wd.sentence_idx_global_pos
	        FROM word_df wd
	        INNER JOIN negativesent sd ON wd.word = sd.word
	    ), temp AS
	    (
	        SELECT s.review_idx_global_pos,
		        w.word,
		        s.sentence
	        FROM word_sentence_sentiment w
	        LEFT JOIN sentences_df s ON w.sentence_idx_global_pos = s.sentence_tokenized_global_pos
	    )
	    SELECT pr_item_sk AS item_sk,
	        sentence AS review_sentence,
	        'NEG' AS sentiment,
	        word as sentiment_word
	    FROM temp 
	    INNER JOIN merged_df ON pr_review_sk = review_idx_global_pos
	    ORDER BY pr_item_sk, review_sentence, sentiment_word
	"""

    result = bc.sql(last_query)

    return result


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
