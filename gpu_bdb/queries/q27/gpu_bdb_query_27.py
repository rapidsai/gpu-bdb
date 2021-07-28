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
import os
import time
import argparse

import spacy
import distributed

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    left_semi_join,
    run_query,
)

from bdb_tools.text import create_sentences_from_reviews, create_words_from_sentences
from bdb_tools.readers import build_reader
from dask.distributed import Client, wait

if os.getenv("DASK_CPU") == "True":
    import dask.dataframe as dask_cudf
else:
    import dask_cudf

# -------- Q27 -----------
q27_pr_item_sk = 10002
EOL_CHAR = "."


def read_tables(config):
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
    return product_reviews_df


def ner_parser(df, col_string, batch_size=256):
    if hasattr(df, "to_pandas"):
        spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(df[col_string], disable=["tagger", "parser"], batch_size=batch_size)
    out = []
    for doc in docs:
        l = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        val = ", "
        l = val.join(l)
        out.append(l)
    df["company_name_list"] = out
    return df


def main(client, config):

    product_reviews_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )
    product_reviews_df = product_reviews_df[
        product_reviews_df.pr_item_sk == q27_pr_item_sk
    ]

    sentences = product_reviews_df.map_partitions(
        create_sentences_from_reviews,
        review_column="pr_review_content",
        end_of_line_char=EOL_CHAR,
    )

    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]

    sentences = sentences.persist()
    wait(sentences)

    # Do the NER
    if hasattr(sentences, "to_dask_dataframe()"):
        sentences = sentences.to_dask_dataframe()
        ner_parsed = sentences.map_partitions(ner_parser, "sentence")
        ner_parsed = dask_cudf.from_dask_dataframe(ner_parsed)
    else:
        ner_parsed = sentences.map_partitions(ner_parser, "sentence")

    ner_parsed = ner_parsed.persist()
    wait(ner_parsed)

    ner_parsed = ner_parsed[ner_parsed.company_name_list != ""]

    # separate NER results into one row per found company
    repeated_names = ner_parsed.map_partitions(
        create_words_from_sentences,
        sentence_column="company_name_list",
        global_position_column="sentence_tokenized_global_pos",
        delimiter="é",
    )

    # recombine
    recombined = repeated_names.merge(
        ner_parsed,
        how="left",
        left_on="sentence_idx_global_pos",
        right_on="sentence_tokenized_global_pos",
    )
    recombined["pr_item_sk"] = q27_pr_item_sk
    recombined = recombined[["review_idx_global_pos", "pr_item_sk", "word", "sentence"]]

    recombined = recombined.persist()
    wait(recombined)

    recombined = recombined.sort_values(
        ["review_idx_global_pos", "pr_item_sk", "word", "sentence"]
    ).persist()

    recombined.columns = ["review_sk", "item_sk", "company_name", "review_sentence"]
    recombined = recombined.persist()
    wait(recombined)
    return recombined


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
