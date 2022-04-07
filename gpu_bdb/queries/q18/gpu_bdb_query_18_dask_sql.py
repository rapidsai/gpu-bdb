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

from nvtx import annotate
import os

from bdb_tools.cluster_startup import attach_to_cluster
import numpy as np

import dask_cudf
import dask.dataframe as dd

from bdb_tools.text import create_sentences_from_reviews, create_words_from_sentences

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q18_utils import (
    find_relevant_reviews,
    q18_startDate,
    q18_endDate,
    EOL_CHAR,
    read_tables
)

from dask.distributed import wait

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c)

    query_1 = f"""
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
                        AND CAST(d.d_date AS DATE) >= DATE '{q18_startDate}'
                        AND CAST(d.d_date AS DATE) <= DATE '{q18_endDate}'
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
    stores_with_regression = c.sql(query_1)

    query_2 = """
        SELECT pr_review_date,
            pr_review_content,
            CAST(pr_review_sk AS INTEGER) AS pr_review_sk
        FROM product_reviews
        WHERE pr_review_content IS NOT NULL
        ORDER BY pr_review_date, pr_review_content, pr_review_sk
    """
    no_nulls = c.sql(query_2)

    targets = (
        stores_with_regression.s_store_name.str.lower()
        .unique()
        .compute()
    )
    
    if isinstance(no_nulls, dask_cudf.DataFrame):
        targets = targets.to_arrow().to_pylist()
    else:
        targets = targets.tolist()
        
    # perssiting because no_nulls is used twice
    no_nulls = no_nulls.persist()

    import cudf
    import pandas as pd
    
    if isinstance(no_nulls, dask_cudf.DataFrame):
        temp_table2_meta_empty_df = cudf.DataFrame(
            {
                "word": ["a"],
                "pr_review_sk": np.ones(1, dtype=np.int64),
                "pr_review_date": ["a"],
            }
        ).head(0)
    else:
        temp_table2_meta_empty_df = pd.DataFrame(
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
    print(no_nulls.compute().head())
    for char in [". ", "? ", "! "]:
        no_nulls["pr_review_content"]  = no_nulls.pr_review_content.str.replace(char, EOL_CHAR, regex=False)

    stores_with_regression["store_ID"] = stores_with_regression.s_store_sk.astype(
        "str"
    ).str.cat(stores_with_regression.s_store_name, sep="_")

    stores_with_regression[
        "s_store_name"
    ] = stores_with_regression.s_store_name.str.lower()

    stores_with_regression = stores_with_regression.persist()
    wait(stores_with_regression)
    c.create_table("stores_with_regression", stores_with_regression, persist=False)
    
    combined = combined.persist()
    wait(combined)
    c.create_table("combined", combined, persist=False)

    query_3 = """
        SELECT store_ID,
            pr_review_date,
            CAST(pr_review_sk AS INTEGER) AS pr_review_sk
        FROM stores_with_regression
        INNER JOIN combined ON s_store_name = word
    """
    temp_table2 = c.sql(query_3)

    c.drop_table("stores_with_regression")
    del stores_with_regression

    c.drop_table("combined")
    del combined

    # REAL QUERY
    print(no_nulls.compute().head())
    sentences = no_nulls.map_partitions(create_sentences_from_reviews)

    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]

    word_df = sentences.map_partitions(
        create_words_from_sentences,
        global_position_column="sentence_tokenized_global_pos",
    )

    # This txt file comes from the official TPCx-BB kit
    # We extracted it from bigbenchqueriesmr.jar
    # Need to pass the absolute path for this txt file
    sentiment_dir = os.path.join(config["data_dir"], "sentiment_files")
    
    if isinstance(word_df, dask_cudf.DataFrame):
        ns_df = dask_cudf.read_csv(os.path.join(sentiment_dir, "negativeSentiment.txt"), names=["sentiment_word"])
    else:
        ns_df = dd.read_csv(os.path.join(sentiment_dir, "negativeSentiment.txt"), names=["sentiment_word"])
    
    c.create_table('sent_df', ns_df, persist=False)

    word_df = word_df.persist()
    wait(word_df)
    c.create_table("word_df", word_df, persist=False)
    
    sentences = sentences.persist()
    wait(sentences)
    c.create_table("sentences", sentences, persist=False)
    
    temp_table2 = temp_table2.persist()
    wait(temp_table2)
    c.create_table("temp_table2", temp_table2, persist=False)

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
    result = c.sql(query_4)

    c.drop_table("word_df")
    del word_df
    c.drop_table("sentences")
    del sentences
    c.drop_table("temp_table2")
    del temp_table2
    return result



@annotate("QUERY18", color="green", domain="gpu-bdb")
def start_run():
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)

if __name__ == "__main__":
    start_run()    
