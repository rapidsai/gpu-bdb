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

if os.getenv("DASK_CPU") == "True":
    import pandas as cudf
else:
    import cudf

EOL_CHAR = "è"


def create_sentences_from_reviews(
    df, review_column="pr_review_content", end_of_line_char=EOL_CHAR
):

    sentences = df[review_column].astype(str).str.split(end_of_line_char)

    # expand the reviews
    tk_cnts = len(sentences) 

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

    cleaned_sentences = df[sentence_column].str.replace(
        "|".join([",", ";", "-", '"']), ""
    ).str.replace(".", " ", regex=False)
    normalized_sentences = cleaned_sentences.str.strip()
    repeat_counts_per_sentence = len(normalized_sentences.str.split(
        delimiter
    ))
    words = normalized_sentences.str.split(delimiter)

    # reassociate with the global position
    global_pos = (
        df[global_position_column]
        .repeat(repeat_counts_per_sentence)
        .reset_index(drop=True)
    )
    out = cudf.DataFrame({"word": words, "sentence_idx_global_pos": global_pos})
    return out
