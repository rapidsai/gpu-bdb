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

import numpy as np
import cupy as cp

import rmm

from cupyx.scipy.sparse import csr_matrix


def rm_punctuations_characters(text_sents):
    """
        * filter punctuation
        * replace multiple spaces with one
        * remove leading spaces and trailing spaces
    """

    filters = [
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "(",
        ")",
        "*",
        "+",
        "-",
        ".",
        "/",
        "\\",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "\~",
        "\t",
        "\\n",
        "'",
        ",",
        "~",
        "—",
    ]

    sents_clean = text_sents.str.replace_multi(filters, " ", regex=False)
    sents_clean = sents_clean.str.normalize_spaces()
    sents_clean = sents_clean.str.strip(" ")
    return sents_clean


def get_ngram(str_series, n, doc_id_sr, token_count_sr, delimiter):
    import cudf
    """
        This returns the ngrams for the string series
         Args:
             str_series(cudf.Series): string series to tokenize
             n(int): gram level to get (1 for unigram , 2 for bigram etc)
             doc_id_sr(cudf.Series): int series containing labels 
             token_count_sr(cudf.Series):  int series containing tokens per doc
             delimiter(string): delimiter to split on

    """    
    ngram_sr = str_series.str.ngrams_tokenize(
        n=n, sep="_", delimiter=delimiter
    )
    
    ### for ngram we have `x-(n-1)`  grams per doc
    ### where x = total number of tokens in the doc
    ### eg: for bigram we have `x-1` bigrams per doc

    token_count_l = token_count_sr - (n - 1)
    doc_id_sr = doc_id_sr.repeat(token_count_l).reset_index(drop=True)
    tokenized_df = cudf.DataFrame()
    tokenized_df["doc_id"] = doc_id_sr
    tokenized_df["token"] = ngram_sr
    return tokenized_df


def create_tokenized_df(str_series, delimiter=" ", ngram_range=(1, 1)):
    """ 
        creates a tokenized df from a string column
        where each row is like [token,doc_id], 
        token = 'string' and doc_id = index from which the word came
        Also returns a  empty doc_id series
    """
    import cudf
    token_count_sr = str_series.str.token_count(delimiter=delimiter)
    doc_id_ar = cp.arange(
        start=0, stop=len(str_series), dtype=np.int32
    )

    doc_id_sr = cudf.Series(doc_id_ar)

    tokenized_df_ls = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngram_ser = get_ngram(str_series, n, doc_id_sr, token_count_sr, delimiter)
        tokenized_df_ls.append(ngram_ser)

    tokenized_df = cudf.concat(tokenized_df_ls)
    tokenized_df = tokenized_df.reset_index(drop=True)

    empty_doc_ids = doc_id_sr[doc_id_sr[token_count_sr == 0]]
    return tokenized_df, empty_doc_ids


def str_token_to_string_id_hash(token_sr, n_features):
    """
        Returns a hashed series of the provided strings
    """
    import cudf
    str_hash_array = rmm.device_array(len(token_sr), dtype=np.uint32)
    token_sr.str.hash(devptr=str_hash_array.device_ctypes_pointer.value)
    # upcasting because we dont support unsigned ints currently
    # see github issue:https://github.com/rapidsai/cudf/issues/2819
    str_hash_array = cp.asarray(cudf.Series(str_hash_array)).astype(np.int64)
    str_hash_array = str_hash_array % n_features
    str_id_sr = cudf.Series(str_hash_array)

    return str_hash_array


def get_count_df(tokenized_df, delimiter=None):
    """
        Returns count of each token in each document
    """
    count_df = (
        tokenized_df[["doc_id", "token"]]
        .groupby(["doc_id", "token"])
        .size()
        .reset_index()
    )
    count_df = count_df.rename({0: "value"})

    return count_df


def create_csr_matrix_from_count_df(count_df, zero_token_docs_ids, n_features):
    """
       count_df = df containg count of hash for each doc
       zero_token_docs_ids = docs that dont have tokens present
       n_features = total number of features
       returns a csr matrix  from the count_df
    """

    data = count_df["value"].values
    indices = count_df["token"].values

    count_df_val_counts = count_df["doc_id"].value_counts().reset_index()
    count_df_val_counts = count_df_val_counts.rename(
        {"doc_id": "token_counts", "index": "doc_id"}
    ).sort_values(by="doc_id")
    indptr = count_df_val_counts["token_counts"].cumsum().values
    indptr = np.pad(indptr, (1, 0), "constant")

    n_rows = len(count_df_val_counts) + len(zero_token_docs_ids)
    if len(zero_token_docs_ids) > 0:
        ValueError("Not Handling empty strings")
        # empty strings can impact etl performence so not handling them
        # only 60 empty strings /1Million
        # handling them should be simple on cpu
        # but dont know how to handle on gpu
    return csr_matrix(
        arg1=(data, indices, indptr), dtype=np.float32, shape=(n_rows, n_features)
    )


def cudf_hashing_vectorizer(
    text_sr,
    n_features=2 ** 25,
    norm=None,
    delimiter=" ",
    preprocessor=rm_punctuations_characters,
    ngram_range=(1, 1),
    lowercase=True,
    alternate_sign=False,
    fill_str="~",
):

    if norm not in [None]:
        raise ValueError("Norm {} is not supported ".format(norm))

    if alternate_sign is not False:
        raise ValueError("alternate_sign  {} is not supported ".format(alternate_sign))

    if preprocessor is not None:
        text_sr = preprocessor(text_sr)

    if lowercase:
        text_sr = text_sr.str.lower()

    tokenized_df, empty_doc_ids = create_tokenized_df(text_sr, delimiter, ngram_range)

    tokenized_df["token"] = str_token_to_string_id_hash(
        tokenized_df["token"], n_features
    )
    count_df = get_count_df(tokenized_df)
    csr = create_csr_matrix_from_count_df(
        count_df, zero_token_docs_ids=empty_doc_ids, n_features=n_features
    )

    return csr
