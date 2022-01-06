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

from bdb_tools.cluster_startup import attach_to_cluster
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
from dask import delayed
import os

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)
from bdb_tools.readers import build_reader
from bdb_tools.cupy_metrics import cupy_precision_score
from sklearn.metrics import roc_auc_score
import cupy as cp


# Logistic Regression params
# solver = "LBFGS" Used by passing `penalty=None` or "l2"
# step_size = 1 Not used
# numCorrections = 10 Not used
iterations = 100
C = 10_000  # reg_lambda = 0 hence C for model is a large value
convergence_tol = 1e-9

wcs_columns = ["wcs_item_sk", "wcs_user_sk"]
items_columns = ["i_item_sk", "i_category", "i_category_id"]
customer_columns = ["c_customer_sk", "c_current_cdemo_sk"]
customer_dem_columns = ["cd_demo_sk", "cd_gender", "cd_education_status"]

def read_tables(data_dir, c, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_ddf = table_reader.read("item", relevant_cols=items_columns, index=False)
    customer_ddf = table_reader.read(
        "customer", relevant_cols=customer_columns, index=False
    )
    customer_dem_ddf = table_reader.read(
        "customer_demographics", relevant_cols=customer_dem_columns, index=False
    )
    wcs_ddf = table_reader.read(
        "web_clickstreams", relevant_cols=wcs_columns, index=False
    )

    c.create_table("web_clickstreams", wcs_ddf, persist=False)
    c.create_table("customer", customer_ddf, persist=False)
    c.create_table("item", item_ddf, persist=False)
    c.create_table("customer_demographics", customer_dem_ddf, persist=False)


def build_and_predict_model(ml_input_df):
    """
    Create a standardized feature matrix X and target array y.
    Returns the model and accuracy statistics
    """
    import cuml
    from cuml.metrics import confusion_matrix

    feature_names = ["college_education", "male"] + [
        "clicks_in_%d" % i for i in range(1, 8)
    ]
    X = ml_input_df[feature_names]
    # Standardize input matrix
    X = (X - X.mean()) / X.std()
    y = ml_input_df["clicks_in_category"]

    model = cuml.LogisticRegression(
        tol=convergence_tol,
        penalty="none",
        solver="qn",
        fit_intercept=True,
        max_iter=iterations,
        C=C,
    )
    model.fit(X, y)
    #
    # Predict and evaluate accuracy
    # (Should be 1.0) at SF-1
    #
    results_dict = {}
    y_pred = model.predict(X)

    results_dict["auc"] = roc_auc_score(y.to_array(), y_pred.to_array())
    results_dict["precision"] = cupy_precision_score(cp.asarray(y), cp.asarray(y_pred))
    results_dict["confusion_matrix"] = confusion_matrix(
        cp.asarray(y, dtype="int32"), cp.asarray(y_pred, dtype="int32")
    )
    results_dict["output_type"] = "supervised"
    return results_dict


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

    query = """
        SELECT
            --wcs_user_sk,
            clicks_in_category,
            CASE WHEN cd_education_status IN ('Advanced Degree', 'College', '4 yr Degree', '2 yr Degree') 
            THEN 1 ELSE 0 END AS college_education,
            CASE WHEN cd_gender = 'M' THEN 1 ELSE 0 END AS male,
            clicks_in_1,
            clicks_in_2,
            clicks_in_3,
            clicks_in_4,
            clicks_in_5,
            clicks_in_6,
            clicks_in_7
        FROM
        ( 
            SELECT 
                wcs_user_sk,
                SUM( CASE WHEN i_category = 'Books' THEN 1 ELSE 0 END) AS clicks_in_category,
                SUM( CASE WHEN i_category_id = 1 THEN 1 ELSE 0 END) AS clicks_in_1,
                SUM( CASE WHEN i_category_id = 2 THEN 1 ELSE 0 END) AS clicks_in_2,
                SUM( CASE WHEN i_category_id = 3 THEN 1 ELSE 0 END) AS clicks_in_3,
                SUM( CASE WHEN i_category_id = 4 THEN 1 ELSE 0 END) AS clicks_in_4,
                SUM( CASE WHEN i_category_id = 5 THEN 1 ELSE 0 END) AS clicks_in_5,
                SUM( CASE WHEN i_category_id = 6 THEN 1 ELSE 0 END) AS clicks_in_6,
                SUM( CASE WHEN i_category_id = 7 THEN 1 ELSE 0 END) AS clicks_in_7
            FROM web_clickstreams
            INNER JOIN item it ON 
            (
                wcs_item_sk = i_item_sk
                AND wcs_user_sk IS NOT NULL
            )
            GROUP BY  wcs_user_sk
        ) q05_user_clicks_in_cat
        INNER JOIN customer ct ON wcs_user_sk = c_customer_sk
        INNER JOIN customer_demographics ON c_current_cdemo_sk = cd_demo_sk
    """

    cust_and_clicks_ddf = c.sql(query)

    cust_and_clicks_ddf = cust_and_clicks_ddf.repartition(npartitions=1)

    # Convert clicks_in_category to a binary label
    cust_and_clicks_ddf["clicks_in_category"] = (
        cust_and_clicks_ddf["clicks_in_category"]
        > cust_and_clicks_ddf["clicks_in_category"].mean()
    ).astype("int64")

    # Converting the dataframe to float64 as cuml logistic reg requires this
    ml_input_df = cust_and_clicks_ddf.astype("float64")

    ml_input_df = ml_input_df.persist()
    wait(ml_input_df)

    ml_tasks = [delayed(build_and_predict_model)(df) for df in ml_input_df.to_delayed()]
    results_dict = client.compute(*ml_tasks, sync=True)

    return results_dict


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
