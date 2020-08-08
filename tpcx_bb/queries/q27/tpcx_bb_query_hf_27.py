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

# Current limitation
### We dont do model based sentence boundary disambiguation
### We get empty sentence in 0.04% of the cases because of it

import time
import rmm
import cupy as cp
import os
import numpy as np
import gc
import distributed
import logging
from dask.distributed import get_worker


from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    left_semi_join,
    run_query,
)

from xbb_tools.text import create_sentences_from_reviews, create_words_from_sentences
from xbb_tools.readers import build_reader
from dask.distributed import Client, wait
from xbb_tools.q27_bert_utils import run_inference_on_df
from xbb_tools.q27_get_review_sentence_utils import get_review_sentence
import torch

#Pytorch/HF imports
from transformers import AutoModelForTokenClassification

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


def create_vocab_table(vocabpath):
    """
        Create Vocabulary tables 
    """
    id2vocab = []
    vocab2id = {}
    with open(vocabpath) as f:
        for index, line in enumerate(f):
            token = line.split()[0]
            id2vocab.append(token)
            vocab2id[token] = index
    return np.array(id2vocab),vocab2id


def load_model(model_path):
    model =  AutoModelForTokenClassification.from_pretrained(model_path)
    model.cuda()
    model.eval()
    return model

def del_model_attribute():
    """
    deletes model attribute
    """
    worker = get_worker()
    if hasattr(worker, 'q27_model'):
        del worker.q27_model
    return   

def run_single_part_workflow(df,model_path):
    """
    This function runs the workflow end2end on a single GPU
    """
       
    w_st = time.time()
    st = time.time()
    worker = get_worker()
    if hasattr(worker, 'q27_model'):
        model = worker.q27_model
    else:
        model = load_model(model_path)
        worker.q27_model=model

    et = time.time()
    logging.warning("Model loading took = {}".format(et-st))

    id2vocab,vocab2id = create_vocab_table(os.path.join(model_path,'vocab.txt'))
    vocab_hash_file = os.path.join(model_path,'vocab-hash.txt')
    st = time.time()
    token_d,prediction_d= run_inference_on_df(df,model,vocab_hash_file)
    et = time.time()
    logging.warning("Inference-E2E took = {}".format(et-st))
    output_d = {}
    st = time.time()
    for seq,pred_label in prediction_d.items():
        if len(pred_label)!=0:
            sen_df = get_review_sentence(token_d[seq],prediction_d[seq],vocab2id,id2vocab)
            review_df = token_d[seq]['df'][['pr_review_sk','pr_item_sk']]
            review_df = review_df.reset_index(drop=False)
            review_df.rename(columns={'index':'input_text_index'},inplace=True)
            output_d[seq] = sen_df.merge(review_df)[['sentence','company','pr_review_sk','pr_item_sk']]
    et = time.time()
    logging.warning("Post Prediction took = {}".format(et-st))
    
    output_df = cudf.concat([o_df for o_df in  output_d.values()])
    w_et = time.time()
    logging.warning("Single part took = {}".format(w_et-w_st))
    return output_df.drop_duplicates()


def main(client, config):

    ### Serializing a pytorch model is slow
    ### Reading from disk write now
    #model_path = '/mnt/weka/vjawa/nlp_model/distilbert-base-en-cased'
    model_path = '/raid/vjawa/torch_ner_q27/transformers/examples/token-classification/distilbert-base-en-cased/'
    product_reviews_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )        
    product_reviews_df = product_reviews_df[product_reviews_df.pr_item_sk == q27_pr_item_sk].persist()

    
    meta_d = {
        'sentence':'',
        'company':'',
        'pr_review_sk': np.ones(1, dtype=np.int64),
        'pr_item_sk': np.ones(1, dtype=np.int64),
    }
    meta_df = cudf.DataFrame(meta_d)
    output_df = product_reviews_df.map_partitions(run_single_part_workflow,model_path,meta=meta_df)
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
    #client.run(rmm.reinitialize,pool_allocator=True,initial_pool_size=10e+9)
    for i in range(3):
        st = time.time()
        run_query(config=config, client=client, query_func=main)
        et = time.time()
        print("Time taken = {}".format(et-st))
        #break
    
   
    #client.run(rmm.reinitialize,pool_allocator=True,initial_pool_size=30e+9)

