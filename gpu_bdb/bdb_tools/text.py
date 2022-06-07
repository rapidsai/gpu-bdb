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

import os

import cudf

import pandas as pd

import numpy as np

EOL_CHAR = "è"


def create_sentences_from_reviews(
    df, review_column="pr_review_content", end_of_line_char=EOL_CHAR,
):
    sentences = df[review_column].str.split(end_of_line_char)
    
    if isinstance(df, cudf.DataFrame):
        out = cudf.DataFrame({"sentence": sentences, "review_idx_global_pos": df.pr_review_sk})
    else:
        out = pd.DataFrame({"sentence": sentences, "review_idx_global_pos": df.pr_review_sk})
    
    out= out.explode("sentence", ignore_index=True)
    out["sentence"] = out.sentence.replace('', np.nan)
    out = out.dropna().reset_index(drop=True)
        
    out["review_idx_global_pos"] = out["review_idx_global_pos"].astype("int32")
    return out


def create_words_from_sentences(
    df,
    sentence_column="sentence",
    global_position_column="sentence_tokenized_global_pos",
    delimiter=" ",
):
    cleaned_sentences = df[sentence_column].str.replace(".", "", regex=False) 
    
    for char in [",", ";", "-", '\"']:
        cleaned_sentences = cleaned_sentences.str.replace(char, "", regex=False) 
        
    normalized_sentences = cleaned_sentences.str.strip() 
    words = normalized_sentences.str.split(delimiter)

    if isinstance(df, cudf.DataFrame):
        out = cudf.DataFrame({"word": words, "sentence_idx_global_pos": df[global_position_column]})
    else:
        out = pd.DataFrame({"word": words, "sentence_idx_global_pos": df[global_position_column]})
    
    out = out.explode("word", ignore_index=True)
    out["word"] = out.word.replace('', np.nan)
    out = out.dropna().reset_index(drop=True)
    
    return out
