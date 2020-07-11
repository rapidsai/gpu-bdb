#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Copyright (c) 2019-2020, BlazingSQL, Inc.
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
import cudf
import numpy as np

from xbb_tools.text import (
    create_sentences_from_reviews,
    create_words_from_sentences
)

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_bsql_query,
)

from tpcx_bb_query_18 import find_relevant_reviews

cli_args = tpcxbb_argparser()

EOL_CHAR = "è"

@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table('store', data_dir + "store/*.parquet")
    bc.create_table('store_sales', data_dir + "store_sales/*.parquet")
    bc.create_table('date_dim', data_dir + "date_dim/*.parquet")
    bc.create_table('product_reviews', data_dir + "product_reviews/*.parquet")

@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query_1 = """
        WITH temp_table1 AS
        (
            SELECT CAST(s.s_store_sk AS INTEGER) AS s_store_sk,
                s.s_store_name ,
                CAST(s.s_store_sk AS VARCHAR) || '_' || s.s_store_name
                    AS store_ID
            FROM store s,
            (
                SELECT temp.ss_store_sk,
                    ((count(temp.x) * SUM(temp.xy) - SUM(temp.x) * SUM(temp.y))
                     / (count(temp.x) * SUM(temp.xx) - SUM(temp.x) * SUM(temp.x))
                     ) AS slope
                FROM
                (
                    SELECT
                        s.ss_store_sk,
                        s.ss_sold_date_sk AS x,
                        CAST( SUM(s.ss_net_paid) AS DOUBLE) AS y,
                        s.ss_sold_date_sk * SUM(s.ss_net_paid) AS xy,
                        s.ss_sold_date_sk * s.ss_sold_date_sk AS xx
                        FROM store_sales s
                        WHERE EXISTS
                    (
                        SELECT * -- d_date_sk
                        FROM date_dim d
                        WHERE s.ss_sold_date_sk = d.d_date_sk
                        AND CAST(d.d_date AS DATE) >= DATE '2001-05-02'
                        AND CAST(d.d_date AS DATE) <= DATE '2001-09-02'
                    )
                        GROUP BY s.ss_store_sk, s.ss_sold_date_sk
                ) temp
                GROUP BY temp.ss_store_sk
            ) regression_analysis
            WHERE slope <= 0 --flat or declining sales
            AND s.s_store_sk = regression_analysis.ss_store_sk
        )
        SELECT * FROM temp_table1
    """
    stores_with_regression = bc.sql(query_1)

    query_2 = """
        SELECT pr_review_date,
            pr_review_content,
            CAST(pr_review_sk AS INTEGER) AS pr_review_sk
        FROM product_reviews
        WHERE pr_review_content IS NOT NULL
    """
    no_nulls = bc.sql(query_2)
    no_nulls = bc.partition(
        no_nulls, by=["pr_review_date", "pr_review_content", "pr_review_sk"]
    )

    targets = (
        stores_with_regression.s_store_name.str.lower().unique().compute().tolist()
    )

    # perssiting because no_nulls is used twice
    no_nulls = no_nulls.persist()

    temp_table2_meta_empty_df = cudf.DataFrame(
        {
            "word": ["a"],
            "pr_review_sk": np.ones(1, dtype=np.int64),
            "pr_review_date": ["a"],
        }
    ).head(0)

    # get relevant reviews
    combined = no_nulls.map_partitions(
        find_relevant_reviews, targets, meta=temp_table2_meta_empty_df,
    )

    no_nulls["pr_review_content"] = no_nulls.pr_review_content.str.replace(
        [". ", "? ", "! "], [EOL_CHAR], regex=False
    )

    stores_with_regression[
        "store_ID"] = stores_with_regression.s_store_sk.astype(
        "str"
    ).str.cat(stores_with_regression.s_store_name, sep="_")

    stores_with_regression[
        "s_store_name"
    ] = stores_with_regression.s_store_name.str.lower()

    bc.create_table('stores_with_regression', stores_with_regression)
    bc.create_table('combined', combined)

    query_3 = """
        SELECT store_ID,
            pr_review_date,
            CAST(pr_review_sk AS INTEGER) AS pr_review_sk
        FROM stores_with_regression
        INNER JOIN combined ON s_store_name = word
    """
    temp_table2 = bc.sql(query_3)

    # REAL QUERY
    sentences = no_nulls.map_partitions(create_sentences_from_reviews)

    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]

    word_df = sentences.map_partitions(
        create_words_from_sentences,
        global_position_column="sentence_tokenized_global_pos",
    )

    bc.create_table('sent_df', os.getcwd() + "/negativeSentiment.txt",
                    names=['sentiment_word'],
                    dtype=['str'])
    bc.create_table('word_df', word_df)
    bc.create_table('sentences', sentences)
    bc.create_table('temp_table2', temp_table2)

    query_4 = """
        WITH sentences_table AS
        (
            select sentence,
                review_idx_global_pos,
                CAST(sentence_tokenized_global_pos AS BIGINT) AS
                 sentence_tokenized_global_pos
            from sentences
        ), negativeSentiment AS
        (
            SELECT DISTINCT sentiment_word AS word
            FROM sent_df
        ), word_sentence_sentiment AS
        (
            SELECT n.word,
                CAST(wd.sentence_idx_global_pos AS BIGINT) AS
                    sentence_idx_global_pos,
                'NEG' AS sentiment
            FROM word_df wd
            INNER JOIN negativeSentiment n ON wd.word = n.word
        ), word_sentence_sentiment_with_sentence_info AS
        (
            SELECT * FROM word_sentence_sentiment
            LEFT JOIN sentences_table
            ON sentence_idx_global_pos = sentence_tokenized_global_pos
        )
        SELECT tt2.store_ID AS s_name,
            tt2.pr_review_date AS r_date,
            wsswsi.sentence AS r_sentence,
            wsswsi.sentiment AS sentiment,
            wsswsi.word AS sentiment_word
        FROM word_sentence_sentiment_with_sentence_info wsswsi
        INNER JOIN temp_table2 tt2
        ON wsswsi.review_idx_global_pos = tt2.pr_review_sk
        ORDER BY s_name, r_date, r_sentence, sentiment_word
    """

    result = bc.sql(query_4)
    return result


if __name__ == "__main__":
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        dask_client=client,
        pool=True,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )
    
    run_bsql_query(
        cli_args=cli_args, client=client, query_func=main
    )
