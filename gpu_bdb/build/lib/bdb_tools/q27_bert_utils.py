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

import cupy as cp
import numpy as np
import torch
import cudf
import time
from torch.utils.dlpack import from_dlpack
from dask.distributed import get_worker


def run_inference_on_df(
    df,
    model,
    vocab_hash_file,
    batchsize=64,
    sequence_len_th_ls=[512, 256, 128, 64, 32, 16, 8],
):
    """
         The function has following steps:
           a. Segregate df based on sequence_length (we do this because inference time is prop to sequence length)
           b. For each part we run tokenization
           c. For each part we run inference using the passed model

        Parameters
        ----------
            df: df to run inference on
            model: model object to run it with
            batchsize:batch size
            sequence_len_th_ls: list of sequences to create batches with

        Returns
        -------
            Returns  token_d,prediction_d  with key=seq_len
    """
    ## Get max sequence_length for a particular review
    df = append_seq_len(df, sequence_len_th_ls, vocab_hash_file)

    ### Tokenize dataframe
    token_d = {}
    max_seq_len = max(sequence_len_th_ls)
    ### Partition each df by sequence length
    for sequence_len, sub_df in get_df_partitioned_by_seq(
        df, sequence_len_th_ls
    ).items():
        if sequence_len == max_seq_len:
            stride = get_stride(max_seq_len)
        else:
            # -2 for padding
            stride = sequence_len - 2
        token_d[sequence_len] = tokenize_text_series(
            sub_df["pr_review_content"],
            sequence_len,
            stride,
            vocab_hash_file=vocab_hash_file,
        )
        token_d[sequence_len]["df"] = sub_df

    del df

    ## Run Inference
    prediction_d = {}
    for seqlen, batch_d in token_d.items():
        prediction_d[seqlen] = run_inference_on_tensor(
            model, batch_d["token_ar"], batch_d["attention_ar"], batchsize
        )
    return token_d, prediction_d


## ----Tokenization Utils----
def append_seq_len(df, sequence_len_ls, vocab_hash_file):
    """
        Appends the sequence length for each review to tokenize too
        The sequence length is the closest max(sequence_len_ls)
        
        Parameters:
        ____________
            df: input dataframe
            max_seq_len: max sequence length to consider
            vocab_hash_file: vocab hash_file to use


    """
    df["input_id"] = cp.arange(0, len(df), dtype=np.int32)
    ### here stride is set to ensure non repeated rows as we want to gather sequence_length
    ### -2 because of padding of special chars
    d = tokenize_text_series(
        df["pr_review_content"],
        max(sequence_len_ls),
        stride=max(sequence_len_ls) - 2,
        vocab_hash_file=vocab_hash_file,
    )

    seq_len_df = get_seq_len_df(d["metadata"], sequence_len_ls)
    seq_len_df = df[["pr_review_sk", "pr_item_sk", "input_id"]].merge(seq_len_df)
    seq_len_df = seq_len_df.groupby("pr_review_sk").sequence_len.max()
    seq_len_df = seq_len_df.reset_index(drop=False)
    df = df.merge(seq_len_df)

    output_columns = ["pr_review_sk", "pr_review_content", "pr_item_sk", "sequence_len"]
    return df[output_columns]


def get_seq_len_df(metadata, sequence_len_ls):
    """
        Returns the sequence_length from the sequence_len_ls to be used 
        for inference
        
        Args:
            metadata: nx3 cupy array(input_id,start_id,stop_id)
            
            sequence_len_ls: list of int sequence_lengths we can have
                (eg:[128,256,512])
        Returns:
            a Cudf Dataframe ([input_id,start_id,stop_id])
    """
    sequence_len_ls = sorted(sequence_len_ls)
    metadata_df = cudf.DataFrame()
    metadata_df["input_id"] = metadata[:, 0]
    metadata_df["start_id"] = metadata[:, 1]
    metadata_df["stop_id"] = metadata[:, 2]

    metadata_df["sequence_len"] = max(sequence_len_ls)
    for output_size in sorted(sequence_len_ls, reverse=True):
        output_flag = metadata_df["stop_id"] < output_size
        metadata_df["sequence_len"][output_flag] = output_size

    return metadata_df


def get_df_partitioned_by_seq(df, sequence_len_ls):
    """
        We get dataframe partitioned by sequences
    """
    sq_part_d = {}
    for s_len in sequence_len_ls:
        subset_df = df[df["sequence_len"] == s_len].reset_index(drop=True)
        sq_part_d[s_len] = subset_df

    return sq_part_d


def tokenize_text_series(text_ser, seq_len, stride, vocab_hash_file):
    """
        This function tokenizes a text series using the bert subword_tokenizer and vocab-hash
        
        Parameters
        __________

        text_ser: Text Series to tokenize
        seq_len: Sequence Length to use (We add to special tokens for ner classification job)
        stride : Stride for the tokenizer
        vocab_hash_file: vocab_hash_file to use (Created using `perfect_hash.py` with compact flag)

        Returns
        _______
         A dictionary with these keys {'token_ar':,'attention_ar':,'metadata':}

    """
    if len(text_ser) == 0:
        return {"token_ar": None, "attention_ar": None, "metadata": None}

    max_num_chars = text_ser.str.len().sum() + 1
    max_rows_tensor = len(text_ser) * 2
    max_length = seq_len - 2

    tokens, attention_masks, metadata = text_ser.str.subword_tokenize(
        vocab_hash_file,
        do_lower=False,
        max_num_strings=max_rows_tensor,
        max_rows_tensor=max_rows_tensor,
        max_num_chars=max_num_chars,
        stride=stride,
        max_length=max_length,
        do_truncate=False,
    )
    del text_ser
    ### reshape metadata into a matrix
    metadata = metadata.reshape(-1, 3)

    tokens = tokens.reshape(-1, max_length)
    output_rows = tokens.shape[0]
    padded_tokens = cp.zeros(shape=(output_rows, seq_len), dtype=np.uint32)

    # Mark sequence start with [CLS] token to mark start of sequence
    padded_tokens[:, 1:-1] = tokens
    padded_tokens[:, 0] = 101

    # Mark end of sequence [SEP]
    seq_end_col = padded_tokens.shape[1] - (padded_tokens[:, ::-1] != 0).argmax(1)
    padded_tokens[cp.arange(padded_tokens.shape[0]), seq_end_col] = 102
    del tokens

    ## Attention mask
    attention_masks = attention_masks.reshape(-1, max_length)
    padded_attention_mask = cp.zeros(shape=(output_rows, seq_len), dtype=np.uint32)
    padded_attention_mask[:, 1:-1] = attention_masks

    # Mark sequence start with 1
    padded_attention_mask[:, 0] = 1

    # Mark sequence end with 1
    padded_attention_mask[cp.arange(padded_attention_mask.shape[0]), seq_end_col] = 1

    del seq_end_col
    del attention_masks

    return {
        "token_ar": padded_tokens,
        "attention_ar": padded_attention_mask,
        "metadata": metadata,
    }


## ----Inference Utils----
def run_inference_on_tensor(model, token_ar, attention_ar, batchsize):
    """
        Runs inference using the model for the given token_ar,attention_ar, batchsize

        Parameters:
        __________
        model: model to use
        token_ar: cupy unsigned int array of shape (n_input_seqs x sequence_length)  containing subword tokens 
        attention_ar: cupy attention unsigned int array of shape (n_input_seqs x sequence_length) containing valid attention mask
        batch_size: batchsize to use

        Returns
        ________
        Predicted tensor of the shape (n_input_seqs x sequence_length)

    """
    if token_ar is None:
        return None

    prediction_ls = []

    batch_st = 0
    total_batches = token_ar.shape[0] // batchsize + 1
    with torch.no_grad():
        token_tensor = from_dlpack(token_ar.astype(np.int32).toDlpack()).long()
        attention_tensor = from_dlpack(attention_ar.astype(np.int32).toDlpack()).long()
        for batch_index in range(0, total_batches):
            batch_st = batch_index * batchsize
            batch_end = min(batch_st + batchsize, token_tensor.shape[0])

            if batch_end == batch_st:
                break

            current_batch_tensor = token_tensor[batch_st:batch_end]
            current_batch_attention = attention_tensor[batch_st:batch_end]

            outputs = model(current_batch_tensor, current_batch_attention)
            prediction_ls.append(outputs[0])

            del current_batch_tensor
            del current_batch_attention

        del token_tensor, attention_tensor

    return torch.cat(prediction_ls).argmax(dim=2)


### Stride Utils
def get_stride(seq_len):
    """
        Stride to use given a sequence length
        Added to ensure we use the same stride across the query
    """
    max_len = seq_len - 2
    stride = int(max_len * 0.5)
    return stride


### Model loading utils
def create_vocab_table(vocabpath):
    """
        Create Vocabulary tables from the vocab.txt file
        
        Parameters:
        ___________
        vocabpath: Path of vocablary file
        Returns:
        ___________
        id2vocab: np.array, dtype=<U5
        vocab2id: dict that maps strings to int
    """
    id2vocab = []
    vocab2id = {}
    with open(vocabpath) as f:
        for index, line in enumerate(f):
            token = line.split()[0]
            id2vocab.append(token)
            vocab2id[token] = index
    return np.array(id2vocab), vocab2id


def load_model(model_path):
    """
        Loads and returns modle from the given model path
    """
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.half()
    model.cuda()
    model.eval()
    return model


def del_model_attribute():
    """
        Deletes model attribute, freeing up memory
    """
    import torch
    import gc

    worker = get_worker()
    if hasattr(worker, "q27_model"):
        del worker.q27_model

    torch.cuda.empty_cache()
    gc.collect()

    return
