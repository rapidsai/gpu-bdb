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

import sys
import os

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
    train_clustering_model
)

from bdb_tools.readers import build_reader

from dask import delayed


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


def read_tables(data_dir, c, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    ss_cols = ["ss_customer_sk", "ss_item_sk"]
    items_cols = ["i_item_sk", "i_category", "i_class_id"]

    ss_ddf = table_reader.read("store_sales", relevant_cols=ss_cols, index=False)
    items_ddf = table_reader.read("item", relevant_cols=items_cols, index=False)

    c.create_table("store_sales", ss_ddf, persist=False)
    c.create_table("item", items_ddf, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

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
            AND i.i_category IN ('{q26_i_category_IN}')
            AND ss.ss_customer_sk IS NOT NULL
        )
        GROUP BY ss.ss_customer_sk
        HAVING count(ss.ss_item_sk) > {q26_count_ss_item_sk}
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
