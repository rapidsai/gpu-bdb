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

from xbb_tools.cluster_startup import attach_to_cluster
import os
import numpy as np
import cupy as cp

from xbb_tools.text import create_sentences_from_reviews, create_words_from_sentences

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

from dask.distributed import wait

# -------- Q18 -----------
q18_startDate = "2001-05-02"
# --+90days
q18_endDate = "2001-09-02"

EOL_CHAR = "è"


def create_found_reshaped_with_global_pos(found, targets):
    """Given the dataframe created by mapping find_targets_in_reviews,
    create a new dataframe in which the nonzero values in each row are exploded
    to get their own row. Each row will contain the word, its mapping in the column order,
    and the pr_review_sk for the review from which it came.

    Having these as two separate functions makes managing dask metadata easier.
    """
    import cudf

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


def find_targets_in_reviews_helper(ddf, targets, str_col_name="pr_review_content"):
    """returns a N x K matrix, where N is the number of rows in ddf that
    contain one of the target words and K is the number of words in targets.
    
    If a target word is found in a review, the value in that row, column
    is non-zero.
    
    At the end, any row with non-zero values is returned.
    
    """
    import cudf
    from cudf._lib.strings import find_multiple

    lowered = ddf[str_col_name].str.lower()

    ## TODO: Do the replace/any in cupy land before going to cuDF
    resdf = cudf.DataFrame(
        cp.asarray(
            find_multiple.find_multiple(lowered._column, targets._column)
        ).reshape(-1, len(targets))
    )

    resdf = resdf.replace([0, -1], [1, 0])
    found_mask = resdf.any(axis=1)
    resdf["pr_review_sk"] = ddf["pr_review_sk"]
    found = resdf.loc[found_mask]
    return create_found_reshaped_with_global_pos(found, targets)


def find_relevant_reviews(df, targets, str_col_name="pr_review_content"):
    """
     This function finds the  reviews containg target stores and returns the 
     relevant reviews
    """
    import cudf

    targets = cudf.Series(targets)
    targets_lower = targets.str.lower()
    reviews_found = find_targets_in_reviews_helper(df, targets_lower)[
        ["word", "pr_review_sk"]
    ]

    combined = reviews_found.merge(
        df[["pr_review_date", "pr_review_sk"]], how="inner", on=["pr_review_sk"]
    )

    return combined


def read_tables(data_dir, bc):
    bc.create_table("store", data_dir + "store/*.parquet")
    bc.create_table("store_sales", data_dir + "store_sales/*.parquet")
    bc.create_table("date_dim", data_dir + "date_dim/*.parquet")
    bc.create_table("product_reviews", data_dir + "product_reviews/*.parquet")


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

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
    stores_with_regression = bc.sql(query_1)

    query_2 = """
        SELECT pr_review_date,
            pr_review_content,
            CAST(pr_review_sk AS INTEGER) AS pr_review_sk
        FROM product_reviews
        WHERE pr_review_content IS NOT NULL
        ORDER BY pr_review_date, pr_review_content, pr_review_sk
    """
    no_nulls = bc.sql(query_2)

    targets = (
        stores_with_regression.s_store_name.str.lower()
        .unique()
        .compute()
        .to_arrow()
        .to_pylist()
    )

    # perssiting because no_nulls is used twice
    no_nulls = no_nulls.persist()

    import cudf

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

    stores_with_regression["store_ID"] = stores_with_regression.s_store_sk.astype(
        "str"
    ).str.cat(stores_with_regression.s_store_name, sep="_")

    stores_with_regression[
        "s_store_name"
    ] = stores_with_regression.s_store_name.str.lower()

    stores_with_regression = stores_with_regression.persist()
    wait(stores_with_regression)
    bc.create_table("stores_with_regression", stores_with_regression)
    
    combined = combined.persist()
    wait(combined)
    bc.create_table("combined", combined)

    query_3 = """
        SELECT store_ID,
            pr_review_date,
            CAST(pr_review_sk AS INTEGER) AS pr_review_sk
        FROM stores_with_regression
        INNER JOIN combined ON s_store_name = word
    """
    temp_table2 = bc.sql(query_3)

    bc.drop_table("stores_with_regression")
    del stores_with_regression

    bc.drop_table("combined")
    del combined

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

    # This txt file comes from the official TPCx-BB kit
    # We extracted it from bigbenchqueriesmr.jar
    # Need to pass the absolute path for this txt file
    sentiment_dir = "/".join(config["data_dir"].split("/")[:-3] + ["sentiment_files"])
    bc.create_table(
        "sent_df",
        sentiment_dir + "/negativeSentiment.txt",
        names=["sentiment_word"],
        dtype=["str"],
    )

    word_df = word_df.persist()
    wait(word_df)
    bc.create_table("word_df", word_df)
    
    sentences = sentences.persist()
    wait(sentences)
    bc.create_table("sentences", sentences)
    
    temp_table2 = temp_table2.persist()
    wait(temp_table2)
    bc.create_table("temp_table2", temp_table2)

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

    bc.drop_table("word_df")
    del word_df
    bc.drop_table("sentences")
    del sentences
    bc.drop_table("temp_table2")
    del temp_table2
    return result


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
