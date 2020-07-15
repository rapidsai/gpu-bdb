#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Copyright (c) 2019-2020, BlazingSQL, Inc.
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

import sys

from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
import os

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_bsql_query,
    train_clustering_model
)

from dask import delayed

cli_args = tpcxbb_argparser()


# -------- Q26 -----------
q26_i_category_IN = "Books"
q26_count_ss_item_sk = 5

N_CLUSTERS = 8
CLUSTER_ITERATIONS = 20
N_ITER = 5


def get_clusters(client, kmeans_input_df):
    import dask_cudf

    ml_tasks = [
        delayed(train_clustering_model)(df, N_CLUSTERS, CLUSTER_ITERATIONS, N_ITER)
        for df in kmeans_input_df.to_delayed()
    ]

    results_dict = client.compute(*ml_tasks, sync=True)

    output = kmeans_input_df.index.to_frame().reset_index(drop=True)

    labels_final = dask_cudf.from_cudf(
        results_dict["cid_labels"], npartitions=output.npartitions
    )
    output["label"] = labels_final.reset_index()[0]

    # Based on CDH6.1 q26-result formatting
    results_dict["cid_labels"] = output
    return results_dict


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table("store_sales", data_dir + "store_sales/*.parquet")
    bc.create_table("item", data_dir + "item/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query = f"""
        SELECT
            ss.ss_customer_sk AS cid,
            count(CASE WHEN i.i_class_id=1  THEN 1 ELSE NULL END) AS id1,
            count(CASE WHEN i.i_class_id=2  THEN 1 ELSE NULL END) AS id2,
            count(CASE WHEN i.i_class_id=3  THEN 1 ELSE NULL END) AS id3,
            count(CASE WHEN i.i_class_id=4  THEN 1 ELSE NULL END) AS id4,
            count(CASE WHEN i.i_class_id=5  THEN 1 ELSE NULL END) AS id5,
            count(CASE WHEN i.i_class_id=6  THEN 1 ELSE NULL END) AS id6,
            count(CASE WHEN i.i_class_id=7  THEN 1 ELSE NULL END) AS id7,
            count(CASE WHEN i.i_class_id=8  THEN 1 ELSE NULL END) AS id8,
            count(CASE WHEN i.i_class_id=9  THEN 1 ELSE NULL END) AS id9,
            count(CASE WHEN i.i_class_id=10 THEN 1 ELSE NULL END) AS id10,
            count(CASE WHEN i.i_class_id=11 THEN 1 ELSE NULL END) AS id11,
            count(CASE WHEN i.i_class_id=12 THEN 1 ELSE NULL END) AS id12,
            count(CASE WHEN i.i_class_id=13 THEN 1 ELSE NULL END) AS id13,
            count(CASE WHEN i.i_class_id=14 THEN 1 ELSE NULL END) AS id14,
            count(CASE WHEN i.i_class_id=15 THEN 1 ELSE NULL END) AS id15
        FROM store_sales ss
        INNER JOIN item i
        ON
        (
            ss.ss_item_sk = i.i_item_sk
            AND i.i_category IN ('{q26_i_category_IN}')
            AND ss.ss_customer_sk IS NOT NULL
        )
        GROUP BY ss.ss_customer_sk
        HAVING count(ss.ss_item_sk) > {q26_count_ss_item_sk}
        ORDER BY cid
    """
    result = bc.sql(query)
    result = result.repartition(npartitions=1)

    # Prepare data for KMeans clustering
    result = result.astype("float64")
    result_ml = result.persist()
    ml_result_dict = get_clusters(client=client, kmeans_input_df=result_ml)
    ml_result_dict['cid_labels'].columns = ["ss_customer_sk", "label"]

    return ml_result_dict


if __name__ == "__main__":
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        dask_client=client,
        pool=True,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )

    run_bsql_query(
        cli_args=cli_args, client=client, query_func=main
    )
