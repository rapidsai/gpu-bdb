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

from xbb_tools.text import (
    create_sentences_from_reviews,
    create_words_from_sentences
)

from xbb_tools.cluster_startup import attach_to_cluster
from dask.distributed import wait
import spacy

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

from dask.distributed import wait

# -------- Q27 -----------
q27_pr_item_sk = 10002
EOL_CHAR = "."


def read_tables(data_dir, bc):
    bc.create_table("product_reviews", data_dir + "/product_reviews/*.parquet")


def ner_parser(df, col_string, batch_size=256):
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


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    import dask_cudf

    query = f"""
        SELECT pr_review_sk, pr_item_sk, pr_review_content
        FROM product_reviews
        WHERE pr_item_sk = {q27_pr_item_sk}
    """
    product_reviews_df = bc.sql(query)

    sentences = product_reviews_df.map_partitions(
        create_sentences_from_reviews,
        review_column="pr_review_content",
        end_of_line_char=EOL_CHAR,
    )

    # need the global position in the sentence tokenized df
    sentences["x"] = 1
    sentences["sentence_tokenized_global_pos"] = sentences.x.cumsum()
    del sentences["x"]
    del product_reviews_df

    # Do the NER
    sentences = sentences.to_dask_dataframe()
    ner_parsed = sentences.map_partitions(ner_parser, "sentence")
    ner_parsed = dask_cudf.from_dask_dataframe(ner_parsed)
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
    del sentences

    # recombine
    repeated_names = repeated_names.persist()
    wait(repeated_names)
    bc.create_table('repeated_names', repeated_names)

    ner_parsed = ner_parsed.persist()
    wait(ner_parsed)
    bc.create_table('ner_parsed', ner_parsed)

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

    bc.drop_table("repeated_names")
    bc.drop_table("ner_parsed")
    del ner_parsed
    del repeated_names
    return recombined


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)

