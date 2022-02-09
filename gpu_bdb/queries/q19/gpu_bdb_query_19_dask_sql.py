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

import dask_cudf

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.text import (
    create_sentences_from_reviews,
    create_words_from_sentences
)

from bdb_tools.q19_utils import (
    q19_returns_dates_IN,
    eol_char,
    read_tables
)

from dask.distributed import wait

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c)

    query = f"""
        WITH dateFilter AS
        (
            -- within the week ending a given date
            SELECT d1.d_date_sk
            FROM date_dim d1, date_dim d2
            WHERE d1.d_week_seq = d2.d_week_seq
            AND CAST(d2.d_date AS DATE) IN (DATE '{q19_returns_dates_IN[0]}',
                                            DATE '{q19_returns_dates_IN[1]}',
                                            DATE '{q19_returns_dates_IN[2]}',
                                            DATE '{q19_returns_dates_IN[3]}')
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
            AND abs( CAST((sr_item_qty-wr_item_qty) AS DOUBLE) /
                ((sr_item_qty + wr_item_qty)/2) ) <= 0.1
        )
        SELECT * FROM extract_sentiment
        ORDER BY pr_item_sk, pr_review_content, pr_review_sk
    """
    merged_df = c.sql(query)

    # second step -- Sentiment Word Extraction
    merged_df["pr_review_sk"] = merged_df["pr_review_sk"].astype("int32")
    merged_df["pr_review_content"] = merged_df.pr_review_content.str.lower()
    merged_df["pr_review_content"] = merged_df.pr_review_content.str.replace(
        [".", "?", "!"], [eol_char], regex=False
    )

    sentences = merged_df.map_partitions(create_sentences_from_reviews)
    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences['sentence_tokenized_global_pos'] = sentences['x'].cumsum()
    del sentences["x"]

    word_df = sentences.map_partitions(
        create_words_from_sentences,
        global_position_column="sentence_tokenized_global_pos",
    )

    # This txt file comes from the official TPCx-BB kit
    # We extracted it from bigbenchqueriesmr.jar
    # Need to pass the absolute path for this txt file
    sentiment_dir = os.path.join(config["data_dir"], "sentiment_files")
    ns_df = dask_cudf.read_csv(os.path.join(sentiment_dir, "negativeSentiment.txt"), names=["sentiment_word"])
    c.create_table('sent_df', ns_df, persist=False)

    sentences = sentences.persist()
    wait(sentences)
    c.create_table('sentences_df', sentences, persist=False)

    word_df = word_df.persist()
    wait(word_df)
    c.create_table('word_df', word_df, persist=False)

    merged_df = merged_df.persist()
    wait(merged_df)
    c.create_table('merged_df', merged_df, persist=False)

    query = """
        WITH negativesent AS
        (
            SELECT distinct sentiment_word
            FROM sent_df
        ), word_sentence_sentiment AS
        (
            SELECT sd.sentiment_word,
                wd.sentence_idx_global_pos
            FROM word_df wd
            INNER JOIN negativesent sd ON wd.word = sd.sentiment_word
        ), temp AS
        (
            SELECT s.review_idx_global_pos,
                w.sentiment_word,
                s.sentence
            FROM word_sentence_sentiment w
            LEFT JOIN sentences_df s
            ON w.sentence_idx_global_pos = s.sentence_tokenized_global_pos
        )
        SELECT pr_item_sk AS item_sk,
            sentence AS review_sentence,
            'NEG' AS sentiment,
            sentiment_word
        FROM temp
        INNER JOIN merged_df ON pr_review_sk = review_idx_global_pos
        ORDER BY pr_item_sk, review_sentence, sentiment_word
    """
    result = c.sql(query)

    c.drop_table("sentences_df")
    del sentences
    c.drop_table("word_df")
    del word_df
    c.drop_table("merged_df")
    del merged_df

    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
