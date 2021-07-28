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

import sys
import os
import glob

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader

import numpy as np
from dask import delayed
import dask
import pandas as pd

if os.getenv("DASK_CPU") == "True":
    import numpy as cp
    import pandas as cudf
    import dask.dataframe as dask_cudf

    import sklearn.linear_model as cuml
    from sklearn.metrics import precision_score, roc_auc_score, confusion_matrix
else:
    import cupy as cp
    import cudf
    import dask_cudf

    import cuml
    from cuml.metrics import roc_auc_score, confusion_matrix
    from bdb_tools.cupy_metrics import cupy_precision_score as precision_score

#
# Query Configuration
#
COLLEGE_ED_STRS = ["Advanced Degree", "College", "4 yr Degree", "2 yr Degree"]
Q05_I_CATEGORY = "Books"

wcs_columns = ["wcs_item_sk", "wcs_user_sk"]
items_columns = ["i_item_sk", "i_category", "i_category_id"]
customer_columns = ["c_customer_sk", "c_current_cdemo_sk"]
customer_dem_columns = ["cd_demo_sk", "cd_gender", "cd_education_status"]

# Logistic Regression params
# solver = "LBFGS" Used by passing `penalty=None` or "l2"
# step_size = 1 Not used
# numCorrections = 10 Not used
iterations = 100
C = 10_000  # reg_lambda = 0 hence C for model is a large value
convergence_tol = 1e-9


def read_tables(config):
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

    return (item_ddf, customer_ddf, customer_dem_ddf)


def build_and_predict_model(ml_input_df):
    """
    Create a standardized feature matrix X and target array y.
    Returns the model and accuracy statistics
    """

    feature_names = ["college_education", "male"] + [
        "clicks_in_%d" % i for i in range(1, 8)
    ]
    X = ml_input_df[feature_names]
    # Standardize input matrix
    X = (X - X.mean()) / X.std()
    y = ml_input_df["clicks_in_category"]

    solver = "lbfgs" if np == cp else "qn"

    model = cuml.LogisticRegression(
        tol=convergence_tol,
        penalty="none",
        solver=solver,
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

    results_dict["auc"] = roc_auc_score(y, y_pred)
    results_dict["precision"] = precision_score(cp.asarray(y), cp.asarray(y_pred))
    results_dict["confusion_matrix"] = confusion_matrix(
        cp.asarray(y, dtype="int32"), cp.asarray(y_pred, dtype="int32")
    )
    results_dict["output_type"] = "supervised"
    return results_dict


def get_groupby_results(file_list, item_df):
    """
        Functionial approach for better scaling
    """

    sum_by_cat_ddf = None
    for fn in file_list:
        wcs_ddf = cudf.read_parquet(fn, columns=wcs_columns)
        wcs_ddf = wcs_ddf[wcs_ddf.wcs_user_sk.notnull()].reset_index(drop=True)
        wcs_ddf = wcs_ddf.merge(
            item_df, left_on="wcs_item_sk", right_on="i_item_sk", how="inner"
        )
        keep_cols = ["wcs_user_sk", "i_category_id", "clicks_in_category"]
        wcs_ddf = wcs_ddf[keep_cols]

        wcs_ddf = cudf.get_dummies(
            wcs_ddf,
            columns=["i_category_id"],
            prefix="clicks_in",
            prefix_sep="_",
            dtype=np.int8,
        )
        keep_cols = ["wcs_user_sk", "clicks_in_category"] + [
            f"clicks_in_{i}" for i in range(1, 8)
        ]
        wcs_ddf = wcs_ddf[keep_cols]

        ### todo: can be shifted downstream to make only 1 groupby call
        grouped_df = wcs_ddf.groupby(["wcs_user_sk"], sort=False, as_index=False).sum()

        if sum_by_cat_ddf is None:
            sum_by_cat_ddf = grouped_df
        else:
            # Roll up to the number of clicks per user
            sum_by_cat_ddf = (
                cudf.concat([sum_by_cat_ddf, grouped_df])
                .groupby("wcs_user_sk", sort=False, as_index=False)
                .sum()
            )

        del grouped_df
        del wcs_ddf

    return sum_by_cat_ddf


def main(client, config):

    item_ddf, customer_ddf, customer_dem_ddf = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    # We want to find clicks in the parameterized category
    # It would be more efficient to translate to a category id, but
    # all of the SQL samples refer to string categories directly We'll
    # call this clicks_in_category to match the names used in SQL
    # examples, though clicks_in_target would be a much better name
    item_ddf["clicks_in_category"] = (
        (item_ddf["i_category"] == Q05_I_CATEGORY)
        .astype(np.int8)
    )
    keep_cols = ["i_item_sk", "i_category_id", "clicks_in_category"]
    item_ddf = item_ddf[keep_cols]

    web_clickstream_flist = glob.glob(os.path.join(config["data_dir"], "web_clickstreams/*.parquet"))
    n_workers = len(client.scheduler_info()["workers"])
    batchsize = len(web_clickstream_flist) // n_workers
    if batchsize < 1:
        batchsize = 1

    chunks = [
        web_clickstream_flist[x : x + batchsize]
        for x in range(0, len(web_clickstream_flist), batchsize)
    ]
    task_ls = [
        delayed(get_groupby_results)(c, item_ddf.to_delayed()[0]) for c in chunks
    ]

    meta_d = {
        "wcs_user_sk": {},
        "clicks_in_category": {},
        "clicks_in_1": {},
        "clicks_in_2": {},
        "clicks_in_3": {},
        "clicks_in_4": {},
        "clicks_in_5": {},
        "clicks_in_6": {},
        "clicks_in_7": {},
    }
    df = pd.DataFrame.from_dict(meta_d, dtype="int64")
    if hasattr(cudf, "from_pandas"):
        df = cudf.from_pandas(df)


    sum_by_cat_ddf = dask_cudf.from_delayed(task_ls, meta=df)
    sum_by_cat_ddf = sum_by_cat_ddf.groupby(["wcs_user_sk"], sort=True).sum()
    sum_by_cat_ddf = sum_by_cat_ddf.reset_index(drop=False)
    #
    # Combine user-level click summaries with customer demographics
    #
    customer_merged_ddf = customer_ddf.merge(
        customer_dem_ddf, left_on="c_current_cdemo_sk", right_on="cd_demo_sk"
    )
    customer_merged_ddf = customer_merged_ddf[
        ["c_customer_sk", "cd_gender", "cd_education_status"]
    ]

    customer_merged_ddf["college_education"] = (
        customer_merged_ddf.cd_education_status.isin(COLLEGE_ED_STRS)
        .astype(np.int64)
        .fillna(0)
    ).reset_index(drop=True)

    customer_merged_ddf["male"] = (
        (customer_merged_ddf["cd_gender"] == "M").astype(np.int64).fillna(0)
    ).reset_index(drop=True)

    cust_and_clicks_ddf = customer_merged_ddf[
        ["c_customer_sk", "college_education", "male"]
    ].merge(sum_by_cat_ddf, left_on="c_customer_sk", right_on="wcs_user_sk")

    keep_cols = ["clicks_in_category", "college_education", "male"] + [
        f"clicks_in_{i}" for i in range(1, 8)
    ]
    cust_and_clicks_ddf = cust_and_clicks_ddf[keep_cols]

    # The ETL step in spark covers everything above this point

    # Convert clicks_in_category to a binary label
    cust_and_clicks_ddf["clicks_in_category"] = (
        (
            cust_and_clicks_ddf["clicks_in_category"]
            > cust_and_clicks_ddf["clicks_in_category"].mean()
        )
        .reset_index(drop=True)
        .astype(np.int64)
    )

    # Converting the dataframe to float64 as cuml logistic reg requires this
    ml_input_df = cust_and_clicks_ddf.astype("float64")

    ml_input_df = ml_input_df.persist()

    ml_tasks = [delayed(build_and_predict_model)(df) for df in ml_input_df.to_delayed()]
    results_dict = client.compute(*ml_tasks, sync=True)

    return results_dict


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
