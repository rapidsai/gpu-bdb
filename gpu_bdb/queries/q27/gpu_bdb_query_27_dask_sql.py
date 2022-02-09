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

from bdb_tools.text import (
    create_sentences_from_reviews,
    create_words_from_sentences
)

from bdb_tools.cluster_startup import attach_to_cluster
from dask.distributed import wait

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q27_utils import (
    ner_parser,
    q27_pr_item_sk,
    EOL_CHAR,
    read_tables
)

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c)

    import dask_cudf

    query = f"""
        SELECT pr_review_sk, pr_item_sk, pr_review_content
        FROM product_reviews
        WHERE pr_item_sk = {q27_pr_item_sk}
    """
    product_reviews_df = c.sql(query)

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
    c.create_table('repeated_names', repeated_names, persist=False)

    ner_parsed = ner_parsed.persist()
    wait(ner_parsed)
    c.create_table('ner_parsed', ner_parsed, persist=False)

    query = f"""
        SELECT review_idx_global_pos as review_sk,
            CAST({q27_pr_item_sk} AS BIGINT) as item_sk,
            word as company_name,
            sentence as review_sentence
        FROM repeated_names left join ner_parsed
        ON sentence_idx_global_pos = sentence_tokenized_global_pos
        ORDER BY review_idx_global_pos, item_sk, word, sentence
    """
    recombined = c.sql(query)

    c.drop_table("repeated_names")
    c.drop_table("ner_parsed")
    del ner_parsed
    del repeated_names
    return recombined


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)

