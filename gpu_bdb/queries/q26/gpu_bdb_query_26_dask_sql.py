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

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
    train_clustering_model
)

from bdb_tools.q26_utils import (
    Q26_CATEGORY,
    Q26_ITEM_COUNT,
    N_CLUSTERS,
    CLUSTER_ITERATIONS,
    N_ITER,
    read_tables
)

from dask import delayed

def get_clusters(client, kmeans_input_df):
    import dask.dataframe as dd
    import dask_cudf
    import cudf
    import pandas as pd
    
    ml_tasks = [
        delayed(train_clustering_model)(df, N_CLUSTERS, CLUSTER_ITERATIONS, N_ITER)
        for df in kmeans_input_df.to_delayed()
    ]

    results_dict = client.compute(*ml_tasks, sync=True)

    output = kmeans_input_df.index.to_frame().reset_index(drop=True)
    
    if isinstance(kmeans_input_df, cudf.DataFrame):
        labels_final = dask_cudf.from_cudf(
            results_dict["cid_labels"], npartitions=output.npartitions
        )
    else:
        labels_final = dd.from_pandas(
            pd.DataFrame(results_dict["cid_labels"]), npartitions=output.npartitions
        )
        
    output["label"] = labels_final.reset_index()[0]

    # Based on CDH6.1 q26-result formatting
    results_dict["cid_labels"] = output
    return results_dict


def main(data_dir, client, c, config):
    benchmark(read_tables, config, c)

    query = f"""
        SELECT
            ss.ss_customer_sk AS cid,
            CAST( count(CASE WHEN i.i_class_id=1  THEN 1 ELSE NULL END) AS DOUBLE ) AS id1,
			CAST( count(CASE WHEN i.i_class_id=2  THEN 1 ELSE NULL END) AS DOUBLE ) AS id2,
			CAST( count(CASE WHEN i.i_class_id=3  THEN 1 ELSE NULL END) AS DOUBLE ) AS id3,
			CAST( count(CASE WHEN i.i_class_id=4  THEN 1 ELSE NULL END) AS DOUBLE ) AS id4,
			CAST( count(CASE WHEN i.i_class_id=5  THEN 1 ELSE NULL END) AS DOUBLE ) AS id5,
			CAST( count(CASE WHEN i.i_class_id=6  THEN 1 ELSE NULL END) AS DOUBLE ) AS id6,
			CAST( count(CASE WHEN i.i_class_id=7  THEN 1 ELSE NULL END) AS DOUBLE ) AS id7,
			CAST( count(CASE WHEN i.i_class_id=8  THEN 1 ELSE NULL END) AS DOUBLE ) AS id8,
			CAST( count(CASE WHEN i.i_class_id=9  THEN 1 ELSE NULL END) AS DOUBLE ) AS id9,
			CAST( count(CASE WHEN i.i_class_id=10 THEN 1 ELSE NULL END) AS DOUBLE ) AS id10,
			CAST( count(CASE WHEN i.i_class_id=11 THEN 1 ELSE NULL END) AS DOUBLE ) AS id11,
			CAST( count(CASE WHEN i.i_class_id=12 THEN 1 ELSE NULL END) AS DOUBLE ) AS id12,
			CAST( count(CASE WHEN i.i_class_id=13 THEN 1 ELSE NULL END) AS DOUBLE ) AS id13,
			CAST( count(CASE WHEN i.i_class_id=14 THEN 1 ELSE NULL END) AS DOUBLE ) AS id14,
			CAST( count(CASE WHEN i.i_class_id=15 THEN 1 ELSE NULL END) AS DOUBLE ) AS id15
        FROM store_sales ss
        INNER JOIN item i
        ON
        (
            ss.ss_item_sk = i.i_item_sk
            AND i.i_category IN ('{Q26_CATEGORY}')
            AND ss.ss_customer_sk IS NOT NULL
        )
        GROUP BY ss.ss_customer_sk
        HAVING count(ss.ss_item_sk) > {Q26_ITEM_COUNT}
        ORDER BY cid
    """
    result = c.sql(query)
    result = result.repartition(npartitions=1)
    result_ml = result.set_index('cid')
    ml_result_dict = get_clusters(client=client, kmeans_input_df=result_ml)
    return ml_result_dict


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
