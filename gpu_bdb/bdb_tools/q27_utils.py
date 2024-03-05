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

import spacy

from bdb_tools.readers import build_reader

q27_pr_item_sk = 10002
EOL_CHAR = "."

def read_tables(config, c=None):
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

    if c:
        c.create_table("product_reviews", product_reviews_df, persist=False)

    return product_reviews_df


def ner_parser(df, col_string, batch_size=256):
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(df[col_string].tolist(), disable=["tagger", "parser"], batch_size=batch_size)
    print(f"docs: {docs}", flush=True)
    out = []
    for doc in docs:
        l = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        val = ", "
        l = val.join(l)
        out.append(l)
    df["company_name_list"] = out
    return df

