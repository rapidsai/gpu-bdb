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

import cupy
import dask

import distributed
import numpy as np
import time
import cupy as cp
import copyreg
import sys, os
import traceback

from distributed import wait
from cuml.feature_extraction.text import HashingVectorizer

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)
from bdb_tools.readers import build_reader

from bdb_tools.q28_utils import post_etl_processing


def read_tables(config):
    ### splitting by row groups for better parallelism
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=True,
    )

    columns = [
        "pr_review_content",
        "pr_review_rating",
        "pr_review_sk",
    ]
    ret = table_reader.read("product_reviews", relevant_cols=columns)
    return ret


def main(client, config):
    q_st = time.time()
    product_reviews_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )
    product_reviews_df = product_reviews_df[
        product_reviews_df["pr_review_content"].notnull()
    ]

    # 90% train/test split
    train_data, test_data = product_reviews_df.random_split([0.9, 0.10])

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    del product_reviews_df

    final_data, acc, prec, cmat = post_etl_processing(
        client=client, train_data=train_data, test_data=test_data
    )
    payload = {
        "df": final_data,
        "acc": acc,
        "prec": prec,
        "cmat": cmat,
        "output_type": "supervised",
    }
    return payload


def register_serialization():
    def serialize_mat_descriptor(m):
        return cp.cupy.cusparse.MatDescriptor.create, ()

    copyreg.pickle(cp.cupy.cusparse.MatDescriptor, serialize_mat_descriptor)


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    import cudf
    from cuml.dask.naive_bayes import MultinomialNB as DistMNB
    from cuml.dask.common.input_utils import DistributedDataHandler
    from cuml.dask.common import to_dask_cudf

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
