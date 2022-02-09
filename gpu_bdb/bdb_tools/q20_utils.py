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
import os

if os.getenv("CPU_ONLY") == 'True':
    import dask.dataframe as dask_cudf
else:
    import dask_cudf

import pandas as pd
from dask import delayed

from bdb_tools.utils import train_clustering_model

from bdb_tools.readers import build_reader

# q20 parameters
N_CLUSTERS = 8
CLUSTER_ITERATIONS = 20
N_ITER = 5

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    store_sales_cols = [
        "ss_customer_sk",
        "ss_ticket_number",
        "ss_item_sk",
        "ss_net_paid",
    ]
    store_returns_cols = [
        "sr_item_sk",
        "sr_customer_sk",
        "sr_ticket_number",
        "sr_return_amt",
    ]

    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    store_returns_df = table_reader.read(
        "store_returns", relevant_cols=store_returns_cols
    )

    if c:
        c.create_table("store_sales", store_sales_df, persist=False)
        c.create_table("store_returns", store_returns_df, persist=False)

    return store_sales_df, store_returns_df


def get_clusters(client, ml_input_df, feature_cols):
    """
    Takes the dask client, kmeans_input_df and feature columns.
    Returns a dictionary matching the output required for q20
    """
    ml_tasks = [
        delayed(train_clustering_model)(df, N_CLUSTERS, CLUSTER_ITERATIONS, N_ITER)
        for df in ml_input_df[feature_cols].to_delayed()
    ]

    results_dict = client.compute(*ml_tasks, sync=True)

    labels = results_dict["cid_labels"]
    
    if hasattr(dask_cudf, "from_cudf"):
        labels_final = dask_cudf.from_cudf(labels, npartitions=ml_input_df.npartitions)
    else:
        labels_final = dask_cudf.from_pandas(pd.DataFrame(labels), npartitions=ml_input_df.npartitions)
         
    
    ml_input_df["label"] = labels_final.reset_index()[0]

    output = ml_input_df[["user_sk", "label"]]

    results_dict["cid_labels"] = output
    return results_dict

