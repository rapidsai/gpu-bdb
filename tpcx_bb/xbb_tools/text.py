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
    df, review_column="pr_review_content", end_of_line_char=EOL_CHAR
):
    import cudf

    sentences = df[review_column].str.tokenize(delimiter=end_of_line_char)

    # expand the reviews
    tk_cnts = df[review_column].str.token_count(delimiter=end_of_line_char)

    # use pr_review_sk as the global position
    # (leaving hardcoded as it's consistent across all queries)
    global_pos = df.pr_review_sk.repeat(tk_cnts).reset_index(drop=True)
    out = cudf.DataFrame({"sentence": sentences, "review_idx_global_pos": global_pos})
    out["review_idx_global_pos"] = out["review_idx_global_pos"].astype("int32")
    return out


def create_words_from_sentences(
    df,
    sentence_column="sentence",
    global_position_column="sentence_tokenized_global_pos",
    delimiter=" ",
):
    import cudf

    cleaned_sentences = df[sentence_column].str.replace(
        [",", ";", "-", '"', "."], ["", "", "", "", " "], regex=False
    )
    normalized_sentences = cleaned_sentences.str.normalize_spaces()
    repeat_counts_per_sentence = normalized_sentences.str.token_count(
        delimiter=delimiter
    )
    words = normalized_sentences.str.tokenize(delimiter=delimiter)

    # reassociate with the global position
    global_pos = (
        df[global_position_column]
        .repeat(repeat_counts_per_sentence)
        .reset_index(drop=True)
    )
    out = cudf.DataFrame({"word": words, "sentence_idx_global_pos": global_pos})
    return out
