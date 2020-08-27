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

# Implimentation Details
# In this query we do NER(Named Entity Resolution) to find  competeter mentions in the review
### The ner model is used is based on Hugging Face's AutoModelForTokenClassification transformer model
### The inference part of workflow to get the token labels are in  q27_bert_utils.run_inference_on_df
### The sentences are gathered using EOL char as `.`
### The details for sentence gathering are at q27_get_review_sentence_utils.get_review_sentence

# Current limitation
### We don't do model based sentence boundary disambiguation
### We get empty sentence in 0.04% of the cases because of it

import rmm
import numpy as np
import os
import logging
import time

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

from xbb_tools.readers import build_reader
from dask.distributed import Client, wait, get_worker

### Query Specific Utils
from xbb_tools.q27_bert_utils import (
    run_inference_on_df,
    load_model,
    create_vocab_table,
    del_model_attribute,
)
from xbb_tools.q27_get_review_sentence_utils import get_review_sentence

# -------- Q27 -----------
q27_pr_item_sk = 10002


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


def run_single_part_workflow(df, model_path):
    """
    This function runs the entire ner workflow end2end on a single GPU
    """
    import cudf

    w_st = time.time()

    worker = get_worker()
    if hasattr(worker, "q27_model"):
        model = worker.q27_model
    else:
        model = load_model(model_path)
        worker.q27_model = model

    id2vocab, vocab2id = create_vocab_table(os.path.join(model_path, "vocab.txt"))
    vocab_hash_file = os.path.join(model_path, "vocab-hash.txt")

    token_d, prediction_d = run_inference_on_df(
        df, model, vocab_hash_file, batchsize=128
    )

    output_d = {}

    for seq, pred_label in prediction_d.items():
        if pred_label is not None:
            sen_df = get_review_sentence(
                token_d[seq], prediction_d[seq], vocab2id, id2vocab
            )
            review_df = token_d[seq]["df"][["pr_review_sk", "pr_item_sk"]]
            review_df = review_df.reset_index(drop=False)
            review_df.rename(columns={"index": "input_text_index"}, inplace=True)
            output_d[seq] = sen_df.merge(review_df)[
                ["pr_review_sk", "pr_item_sk", "company_name", "review_sentence"]
            ]

    del token_d, prediction_d

    output_df = cudf.concat([o_df for o_df in output_d.values()])
    output_df.rename(
        columns={"pr_review_sk": "review_sk", "pr_item_sk": "item_sk"}, inplace=True
    )

    w_et = time.time()
    logging.warning("Single part took = {}".format(w_et - w_st))
    return output_df.drop_duplicates()


def main(client, config):

    import cudf

    model_path = os.path.join(config["data_dir"], "../../q27_model_dir")
    product_reviews_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )
    product_reviews_df = product_reviews_df[
        product_reviews_df.pr_item_sk == q27_pr_item_sk
    ].persist()

    meta_d = {
        "review_sk": np.ones(1, dtype=np.int64),
        "item_sk": np.ones(1, dtype=np.int64),
        "company_name": "",
        "review_sentence": "",
    }
    meta_df = cudf.DataFrame(meta_d)
    output_df = product_reviews_df.map_partitions(
        run_single_part_workflow, model_path, meta=meta_df
    )
    output_df = output_df.persist()
    wait(output_df)
    client.run(del_model_attribute)
    return output_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=14e9)
    run_query(config=config, client=client, query_func=main)
