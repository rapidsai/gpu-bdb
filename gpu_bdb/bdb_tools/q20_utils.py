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

import dask_cudf

from dask import delayed

from bdb_tools.utils import train_clustering_model

# q20 parameters
N_CLUSTERS = 8
CLUSTER_ITERATIONS = 20
N_ITER = 5

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

    labels_final = dask_cudf.from_cudf(labels, npartitions=ml_input_df.npartitions)
    ml_input_df["label"] = labels_final.reset_index()[0]

    output = ml_input_df[["user_sk", "label"]]

    results_dict["cid_labels"] = output
    return results_dict

