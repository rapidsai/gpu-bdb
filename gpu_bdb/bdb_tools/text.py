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

EOL_CHAR = "è"


def create_sentences_from_reviews(
    df, review_column="pr_review_content", end_of_line_char=EOL_CHAR,
):
    import pandas as pd
    import cudf
    import numpy as np
    
    if isinstance(df, cudf.DataFrame):
        sentences = df[review_column].str.tokenize(delimiter=end_of_line_char)   
        tk_cnts = df[review_column].str.token_count(delimiter=end_of_line_char)
    else:
        sentences = df[review_column].str.split(end_of_line_char)
        tk_cnts = sentences.str.len()
        sentences = sentences.explode(ignore_index=True)
        
        
    # use pr_review_sk as the global position
    # (leaving hardcoded as it's consistent across all queries)
    global_pos = df.pr_review_sk.repeat(tk_cnts).reset_index(drop=True)
    
    if isinstance(df, cudf.DataFrame):
        out = cudf.DataFrame({"sentence": sentences, "review_idx_global_pos": global_pos})
    else:
        out = pd.DataFrame({"sentence": sentences, "review_idx_global_pos": global_pos})
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
   
    import pandas as pd
    import cudf
    
    cleaned_sentences = df[sentence_column].str.replace(".", "", regex=False) 
    
    for char in [",", ";", "-", '\"']:
        cleaned_sentences = cleaned_sentences.str.replace(char, "", regex=False) 
    
    if isinstance(df, cudf.DataFrame):
        normalized_sentences = cleaned_sentences.str.normalize_spaces()
        repeat_counts_per_sentence = normalized_sentences.str.token_count(
            delimiter=delimiter
        )
        words = normalized_sentences.str.tokenize(delimiter=delimiter)
    else:
        normalized_sentences = cleaned_sentences.str.strip() 
        words = normalized_sentences.str.split(delimiter)
        repeat_counts_per_sentence = words.str.len()
        words= words.explode(ignore_index=True)
        
    # reassociate with the global position
    global_pos = (
        df[global_position_column]
        .repeat(repeat_counts_per_sentence)
        .reset_index(drop=True)
    )
    if isinstance(df, cudf.DataFrame):
        out = cudf.DataFrame({"word": words, "sentence_idx_global_pos": global_pos})
    else:
        out = pd.DataFrame({"word": words, "sentence_idx_global_pos": global_pos})
        out["word"] = out.sentence.replace('', np.nan)
        out = out.dropna().reset_index(drop=True)
    return out
