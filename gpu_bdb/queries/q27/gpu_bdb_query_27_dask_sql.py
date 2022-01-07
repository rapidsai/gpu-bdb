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
import os

from bdb_tools.text import (
    create_sentences_from_reviews,
    create_words_from_sentences
)

from bdb_tools.cluster_startup import attach_to_cluster
from dask.distributed import wait
import spacy

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader

from dask.distributed import wait


# -------- Q27 -----------
q27_pr_item_sk = 10002
EOL_CHAR = "."


def read_tables(data_dir, bc, config):
    ### splitting by row groups for better parallelism
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=True,
    )
    product_reviews_cols = ["pr_item_sk", "pr_review_content", "pr_review_sk"]
    product_reviews_df = table_reader.read(
        "product_reviews", relevant_cols=product_reviews_cols
    )

    bc.create_table("product_reviews", product_reviews_df, persist=False)

    # bc.create_table("product_reviews", os.path.join(data_dir, "product_reviews/*.parquet"))


def ner_parser(df, col_string, batch_size=256):
    #import spacy
    #print(df)
    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(df[col_string], disable=["tagger", "parser"], batch_size=batch_size)
    #print(docs)
    out = []
    for doc in docs:
        #print(doc)
        l = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        #print(l)
        val = ", "
        l = val.join(l)
        out.append(l)
        #print(out)
    df["company_name_list"] = out
    return df


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, config, dask_profile=config["dask_profile"])

    import dask.dataframe as dask_cudf
    import pandas as cudf
    import numpy as np
    
    query = f"""
        SELECT pr_review_sk, pr_item_sk, pr_review_content
        FROM product_reviews
        WHERE pr_item_sk = {q27_pr_item_sk}
    """
    product_reviews_df = bc.sql(query)
    #print(product_reviews_df.index.compute().is_unique)
    

    sentences = product_reviews_df.map_partitions(
        create_sentences_from_reviews,
        review_column="pr_review_content",
        end_of_line_char=EOL_CHAR,
    )
    #print(sentences.compute())
    
    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]
    del product_reviews_df
    
    
    # Do the NER
    ner_parsed = sentences.map_partitions(ner_parser, "sentence")
    ner_parsed = ner_parsed.persist()
    wait(ner_parsed)
    #print(ner_parsed.compute())
    ner_parsed = ner_parsed[ner_parsed.company_name_list != ""]

    # separate NER results into one row per found company
    repeated_names = ner_parsed.map_partitions(
        create_words_from_sentences,
        sentence_column="company_name_list",
        global_position_column="sentence_tokenized_global_pos",
        delimiter="é",
    )
    del sentences

    # recombine
    repeated_names = repeated_names.persist()
    wait(repeated_names)
    bc.create_table('repeated_names', repeated_names, persist=False)

    ner_parsed = ner_parsed.persist()
    wait(ner_parsed)
    bc.create_table('ner_parsed', ner_parsed, persist=False)

    query = f"""
        SELECT review_idx_global_pos as review_sk,
            CAST({q27_pr_item_sk} AS BIGINT) as item_sk,
            word as company_name,
            sentence as review_sentence
        FROM repeated_names left join ner_parsed
        ON sentence_idx_global_pos = sentence_tokenized_global_pos
        ORDER BY review_idx_global_pos, item_sk, word, sentence
    """
    recombined = bc.sql(query)
    #print(recombined.compute())

    bc.drop_table("repeated_names")
    bc.drop_table("ner_parsed")
    del ner_parsed
    del repeated_names
    recombined.compute()
    return recombined


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)

