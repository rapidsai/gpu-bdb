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


import cudf
import torch
from numba import cuda
from .q27_bert_utils import get_stride
import cupy as cp
import logging
import numpy as np


def get_review_sentence(
    tokenized_d, predicted_label_t, vocab2id, id2vocab, org_labels=[2, 5]
):
    """
        Given a tokenized_d and predicted_label_t
        We return the  sentences that contain labels for either 5 or 2

        #### Detailed Workflow ####

        ## This function contains the review sentence gathering part of workflow
        # Given a tokenized_d and predicted_label_t
        # we gather the  sentences that contain labels for it

        ### The workflow is as follows:
        ### First we get org_df which contains the location of everything predicted as an organization
        ### Second we get all the sentence boundaries
        ### Third, we find the sentences that correspond to our predicted label
        ### Fourth, From that org_sentences_table, we create a sentence matrix which contains all tokens that will go in that sentence
        ### Fifth, We convert that matrix into strings (This step happens on CPU)

    """
    seq_len = tokenized_d["token_ar"].shape[1]
    stride = get_stride(seq_len)

    metadata_df = cudf.DataFrame()
    metadata_df = cudf.DataFrame()
    metadata_df["input_text_index"] = tokenized_d["metadata"][:, 0]
    ## +1 for clx class
    metadata_df["start_index"] = tokenized_d["metadata"][:, 1] + 1
    metadata_df["stop_index"] = tokenized_d["metadata"][:, 2] + 1
    metadata_df["seq_row"] = cp.arange(len(metadata_df))

    pr_label_f = None
    for label in org_labels:
        if pr_label_f is None:
            pr_label_f = predicted_label_t == label
        else:
            pr_label_f = pr_label_f | (predicted_label_t == label)

    ## Step1: Get ORG
    org_df = get_org_df(pr_label_f, metadata_df, seq_len)
    ### Because we have repeations in our boundaries we
    ### create a valid region boundary to prevent copying
    valid_region = (seq_len - 2) - stride + 1
    ### This gives us all the valid sentence boundaries

    ### Step2: Get Sentence Boundary
    sentence_boundary_df = get_sentence_boundaries(
        metadata_df,
        tokenized_d["token_ar"],
        stride=stride,
        fs_index_ls=[vocab2id["."], vocab2id["##."]],
    )

    ## Step3: df contains the sentences that intersect with org
    ## Sentence containing ORG
    org_senten_df = get_org_sentences(sentence_boundary_df, org_df)
    org_senten_df = org_senten_df.reset_index(drop=False)

    ## Step4:Flatten these sentences and add them to the output matrix
    output_mat = cp.zeros(
        shape=(len(org_senten_df["org_seq_row"]), 1024 * 2), dtype=np.int32
    )
    label_ar = cp.zeros(shape=(len(org_senten_df["org_seq_row"]), 1), dtype=np.int32)

    input_mat = tokenized_d["token_ar"]

    l_r_ar = org_senten_df["l_fs_seq_row"]
    l_c_ar = org_senten_df["l_fs_seq_col"]

    r_r_ar = org_senten_df["r_fs_seq_row"]
    r_c_ar = org_senten_df["r_fs_seq_col"]

    o_r_ar = org_senten_df["org_seq_row"]
    o_c_ar = org_senten_df["org_seq_col"]

    get_output_sen_word_kernel.forall(len(l_r_ar))(
        l_r_ar,
        l_c_ar,
        r_r_ar,
        r_c_ar,
        o_r_ar,
        o_c_ar,
        valid_region,
        input_mat,
        output_mat,
        label_ar,
    )

    output_mat = cp.asnumpy(output_mat)
    label_ar = cp.asnumpy(label_ar).flatten()

    ### Step5: Detokenize the matrix

    ### CPU logic to gather sentences begins here
    sen_ls = []
    target_ls = []
    for row, t_num in zip(output_mat, label_ar):
        s, t = convert_to_sentence(row, t_num, id2vocab)
        sen_ls.append(s)
        target_ls.append(t)

    df = cudf.DataFrame()
    df["review_sentence"] = cudf.Series(sen_ls, dtype="str")
    df["company_name"] = cudf.Series(target_ls, dtype="str")
    df["input_text_index"] = org_senten_df["input_text_index"]

    return df


def get_sentence_boundaries(metadata_df, token_ar, stride, fs_index_ls):
    """
        Given token array and meta-data we create sentence boundaries 
        We consider a sentence boundary as one which is at eol-chars (`##.`,`.`) or start/end of a review
    """
    seq_len = token_ar.shape[1]

    fullstop_flag = None
    for fs_token_idx in fs_index_ls:
        if fullstop_flag is None:
            fullstop_flag = token_ar == fs_token_idx
        else:
            fullstop_flag = (fullstop_flag) | (token_ar == fs_token_idx)

    fullstop_row, fullstop_col = cp.nonzero(fullstop_flag)

    min_row_df = (
        metadata_df.groupby("input_text_index").seq_row.min().reset_index(drop=False)
    )
    min_row_df.rename(columns={"seq_row": "min_row"}, inplace=True)
    max_row_df = (
        metadata_df.groupby("input_text_index").seq_row.max().reset_index(drop=False)
    )
    max_row_df.rename(columns={"seq_row": "max_row"}, inplace=True)

    metadata_df = metadata_df.merge(min_row_df).merge(max_row_df)

    ### Can filter to only sequences that have the org
    ## if below becomes a bottleneck

    fullstop_df = cudf.DataFrame()
    fullstop_df["seq_row"] = cudf.Series(fullstop_row)
    fullstop_df["fs_seq_col"] = cudf.Series(fullstop_col)
    fullstop_df = fullstop_df.merge(metadata_df)

    fullstop_df.rename(columns={"seq_row": "fs_seq_row"}, inplace=True)

    first_row_df = cudf.DataFrame()
    first_row_df["input_text_index"] = min_row_df["input_text_index"]
    first_row_df["fs_seq_row"] = min_row_df["min_row"]
    first_row_df["fs_seq_col"] = 1
    first_row_df["min_row"] = min_row_df["min_row"]
    first_row_df = first_row_df.merge(max_row_df[["input_text_index", "max_row"]])

    last_row_df = cudf.DataFrame()
    last_row_df["input_text_index"] = max_row_df["input_text_index"]
    last_row_df["fs_seq_row"] = max_row_df["max_row"]
    last_row_df["fs_seq_col"] = seq_len - 1
    last_row_df["max_row"] = max_row_df["max_row"]
    last_row_df = last_row_df.merge(min_row_df[["input_text_index", "min_row"]])

    fullstop_df = cudf.concat([fullstop_df, first_row_df, last_row_df])

    ## -2-> for padding
    valid_region = (seq_len - 2) - stride + 1

    ### only keep sentences in the valid_region
    valid_flag = fullstop_df["fs_seq_col"] < valid_region
    valid_flag = valid_flag | (fullstop_df["fs_seq_row"] == fullstop_df["max_row"])
    fullstop_df = fullstop_df[valid_flag]

    fullstop_df["flat_loc_fs"] = (
        fullstop_df["fs_seq_row"] * seq_len + fullstop_df["fs_seq_col"]
    )

    return fullstop_df[["input_text_index", "fs_seq_row", "fs_seq_col", "flat_loc_fs"]]


def get_org_sentences(sentence_boundary_df, org_df):
    """
        Given the sentence_boundary_df and org_df,  returns the nearest sentence boundries that contain org.

        Returns a org_senten_df 
    """
    merged_df = sentence_boundary_df.merge(org_df, on="input_text_index")
    merged_df["left_loc"] = merged_df["flat_loc_org"] - merged_df["flat_loc_fs"]
    merged_df["right_loc"] = merged_df["flat_loc_fs"] - merged_df["flat_loc_org"]

    ### Better way to get the closeset row/col maybe
    valid_left_loc = (
        merged_df[merged_df["left_loc"] >= 0]
        .sort_values(by=["flat_loc_org", "left_loc"])
        .groupby("flat_loc_org")
        .nth(0)
    )

    cols_2_keep = [
        "input_text_index",
        "fs_seq_row",
        "fs_seq_col",
        "org_seq_row",
        "org_seq_col",
    ]
    valid_left_loc = valid_left_loc[cols_2_keep]
    valid_left_loc.rename(
        columns={"fs_seq_row": "l_fs_seq_row", "fs_seq_col": "l_fs_seq_col"},
        inplace=True,
    )
    valid_left_loc = valid_left_loc.reset_index(drop=False)

    ### Better way to get the closeset row/col maybe
    valid_right_loc = (
        merged_df[merged_df["right_loc"] > 0]
        .sort_values(by=["flat_loc_org", "right_loc"])
        .groupby("flat_loc_org")
        .nth(0)
    )
    valid_right_loc.rename(
        columns={"fs_seq_row": "r_fs_seq_row", "fs_seq_col": "r_fs_seq_col"},
        inplace=True,
    )

    valid_right_loc = valid_right_loc[["r_fs_seq_row", "r_fs_seq_col"]].reset_index(
        drop=False
    )

    valid_df = valid_left_loc.merge(valid_right_loc)
    valid_df = valid_df.set_index(["flat_loc_org"])

    return valid_df


@cuda.jit
def get_output_sen_word_kernel(
    start_r_ar,
    start_c_ar,
    end_r_ar,
    end_c_ar,
    t_row_ar,
    t_col_ar,
    valid_region,
    mat,
    output_mat,
    label_ar,
):

    """
        Fills the output_matrix and label_ar for each review sentence
    """

    rnum = cuda.grid(1)

    if rnum < (start_r_ar.size):  # boundary guard
        start_r, start_c = start_r_ar[rnum], start_c_ar[rnum]
        end_r, end_c = end_r_ar[rnum], end_c_ar[rnum]
        t_row, t_col = t_row_ar[rnum], t_col_ar[rnum]

        i = 0
        for curr_r in range(start_r, end_r + 1):
            if curr_r == start_r:
                col_loop_s = start_c
            else:
                col_loop_s = 1

            if curr_r == end_r:
                col_loop_e = end_c
            else:
                col_loop_e = valid_region

            for curr_c in range(col_loop_s, col_loop_e):
                token = mat[curr_r][curr_c]
                if token != 0:
                    output_mat[rnum][i] = token
                    i += 1
                    if (curr_r == t_row) and (curr_c == t_col):
                        label_ar[rnum] = i

    return


### CPU part of workflow
def convert_to_sentence(row, target_index, id2vocab):
    """
        Given a row of token_ids , we convert to a sentence
        We also combine subtokens back to get back the input sentence
    """
    row = row[row != 0]
    output_ls = []
    tr_index = -1
    row = id2vocab[row]
    for t_num, token in enumerate(row):
        if t_num == target_index:
            tr_index = len(output_ls) - 1
        ## We anyways skip the first full-stop and we dont want to combine that
        ### eg: test. new sen ---tokenized-> test ##. new sen
        ### we only will want to capture "new sen"
        if len(output_ls) > 0 and token.startswith("##"):
            output_ls[-1] += token[2:]
        else:
            output_ls.append(token)

    if output_ls[0] in [".", "##."]:
        output_sen = " ".join(output_ls[1:])
    else:
        output_sen = " ".join(output_ls)

    return output_sen, output_ls[tr_index]


def get_org_df(pr_label_f, metadata_df, seq_len):
    """
        Returns the org_df given pr_label_f,metadata_df,
    """
    org_r, org_c = torch.nonzero(pr_label_f, as_tuple=True)
    org_df = cudf.DataFrame()
    org_df["seq_row"] = cudf.Series(org_r)
    org_df["org_seq_col"] = cudf.Series(org_c)
    org_df = org_df.merge(metadata_df)
    org_df = org_df.rename(columns={"seq_row": "org_seq_row"})

    org_df["flat_loc_org"] = org_df["org_seq_row"] * seq_len + org_df["org_seq_col"]
    ### Trim overlapping and invalid predictions
    flag = (org_df["org_seq_col"] >= org_df["start_index"]) & (
        org_df["org_seq_col"] <= org_df["stop_index"]
    )
    org_df = org_df[flag]

    return org_df[["org_seq_row", "org_seq_col", "input_text_index", "flat_loc_org"]]
