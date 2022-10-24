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

import numpy as np
import cupy as cp
import cupy

import pandas as pd
import cudf

import dask
import dask_cudf

from cuml.feature_extraction.text import HashingVectorizer
from cuml.dask.naive_bayes import MultinomialNB as DistMNB
from cuml.dask.common import to_dask_cudf

from sklearn.feature_extraction.text import HashingVectorizer as SKHashingVectorizer
from sklearn.naive_bayes import MultinomialNB as MultNB

from dask_ml.wrappers import ParallelPostFit

import scipy

from distributed import wait

from uuid import uuid1

from bdb_tools.readers import build_reader

from cuml.dask.common.part_utils import _extract_partitions

N_FEATURES = 2 ** 23  # Spark is doing 2^20
ngram_range = (1, 2)
norm = None
alternate_sign = False

def read_tables(config, c=None):
    ### splitting by row groups for better parallelism
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=True,
        backend=config["backend"]
    )

    columns = [
        "pr_review_content",
        "pr_review_rating",
        "pr_review_sk",
    ]
    pr_df = table_reader.read("product_reviews", relevant_cols=columns)

    if c:
        c.create_table("product_reviews", pr_df, persist=False)

    return pr_df


def hashing_vectorizer(x):

    if isinstance(x, cudf.Series):
        vectorizer = HashingVectorizer
        preprocessor = lambda s:s.str.lower()
    else:
        vectorizer = SKHashingVectorizer
        preprocessor = lambda s:s.lower()

    vec = vectorizer(
        n_features=N_FEATURES,
        alternate_sign=alternate_sign,
        ngram_range=ngram_range,
        norm=norm,
        preprocessor=preprocessor
    )

    return vec.fit_transform(x)


def map_labels(ser):

    if isinstance(ser, cudf.Series):
        output_ser = cudf.Series(cudf.core.column.full(size=len(ser), fill_value=2, dtype=np.int32))
    else:
        output_ser = pd.Series(2, index=ser.index, dtype=np.int32)

    zero_flag = (ser==1) | (ser==2)
    output_ser.loc[zero_flag]=0

    three_flag = (ser==3)
    output_ser.loc[three_flag]=1

    return output_ser


def build_features(t):

    if isinstance(t, dask_cudf.DataFrame):
        meta_arr = dask.array.from_array(
            cp.sparse.csr_matrix(cp.zeros(1, dtype=np.float32))
        )
    else:
        meta_arr = dask.array.from_array(
            scipy.sparse.csr_matrix(np.zeros(1, dtype=np.float32))
        )

    X = t["pr_review_content"]
    X = X.map_partitions(
        hashing_vectorizer,
        meta=meta_arr,
    )

    X = X.astype(np.float32).persist()
    X.compute_chunk_sizes()

    return X


def build_labels(reviews_df):
    y = reviews_df["pr_review_rating"].map_partitions(map_labels)

    if isinstance(reviews_df, dask_cudf.DataFrame):
        y = y.map_partitions(lambda x: cp.asarray(x, np.int32)).persist()
        y._meta = cp.array(y._meta)
    else:
        y = y.map_partitions(lambda x: np.asarray(x, np.int32)).persist()

    y.compute_chunk_sizes()

    return y


def categoricalize(num_sr):

    return (num_sr
            .astype(str)
            .str.replace("0", "NEG")
            .str.replace("1", "NEUT")
            .str.replace("2", "POS")  
    )


def sum_tp_fp(y_y_pred, nclasses):

    y, y_pred = y_y_pred

    res = np.zeros((nclasses, 2), order="F", like=y)

    for i in range(nclasses):
        if isinstance(y, cp.ndarray):
            pos_pred_ix = cp.where(y_pred == i)[0]
        else:
            pos_pred_ix = np.where(y_pred == i)[0]

        # short circuit
        if len(pos_pred_ix) == 0:
            res[i] = 0
            break

        tp_sum = (y_pred[pos_pred_ix] == y[pos_pred_ix]).sum()
        fp_sum = (y_pred[pos_pred_ix] != y[pos_pred_ix]).sum()
        res[i][0] = tp_sum
        res[i][1] = fp_sum

    return res

def precision_score(client, y, y_pred, average="binary"):

    if isinstance(y._meta, cp.ndarray):
        nclasses = len(cp.unique(y.map_blocks(lambda x: cp.unique(x)).compute()))
    else:
        nclasses = len(np.unique(y.map_blocks(lambda x: np.unique(x)).compute()))

    if average == "binary" and nclasses > 2:
        raise ValueError

    if nclasses < 2:
        raise ValueError("Single class precision is not yet supported")

    gpu_futures = client.sync(_extract_partitions, [y, y_pred], client)

    precision_scores = client.compute(
        [
            client.submit(sum_tp_fp, part, nclasses, workers=[worker])
            for worker, part in gpu_futures
        ],
        sync=True,
    )

    res = np.zeros((nclasses, 2), order="F", like=y._meta)

    for i in precision_scores:
        res += i

    if average == "binary" or average == "macro":
        prec = np.zeros(nclasses, like=y._meta)

        for i in range(nclasses):
            tp_sum, fp_sum = res[i]
            prec[i] = (tp_sum / (tp_sum + fp_sum)).item()

        if average == "binary":
            return prec[nclasses - 1].item()
        else:
            return prec.mean().item()
    else:
        if isinstance(y._meta, cp.ndarray):
            global_tp = cp.sum(res[:, 0])
            global_fp = cp.sum(res[:, 1])
        else:
            global_tp = np.sum(res[:, 0])
            global_fp = np.sum(res[:, 1])

        return global_tp / (global_tp + global_fp).item()


def local_cm(y_y_pred, unique_labels, sample_weight):

    y_true, y_pred = y_y_pred
    labels = unique_labels

    n_labels = labels.size

    # Assume labels are monotonically increasing for now.

    # intersect y_pred, y_true with labels, eliminate items not in labels
    if isinstance(y_true, cp.ndarray):
        ind = cp.logical_and(y_pred < n_labels, y_true < n_labels)
    else:
        ind = np.logical_and(y_pred < n_labels, y_true < n_labels)

    y_pred = y_pred[ind]
    y_true = y_true[ind]

    if isinstance(y_true, cp.ndarray):
        if sample_weight is None:
            sample_weight = cp.ones(y_true.shape[0], dtype=np.int64)
        else:
            sample_weight = cp.asarray(sample_weight)
    else:
        if sample_weight is None:
            sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
        else:
            sample_weight = np.asarray(sample_weight)

    sample_weight = sample_weight[ind]

    if isinstance(y_true, cp.ndarray):
        cm = cupy.sparse.coo_matrix(
            (sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=np.float32,
        ).toarray()

        return cp.nan_to_num(cm)
    else:
        cm = scipy.sparse.coo_matrix(
            (sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=np.float32,
        ).toarray()

        return np.nan_to_num(cm)


def confusion_matrix(client, y_true, y_pred, normalize=None, sample_weight=None):

    if isinstance(y_true._meta, cp.ndarray):
        unique_classes = cp.unique(y_true.map_blocks(lambda x: cp.unique(x)).compute())
    else:
        unique_classes = np.unique(y_true.map_blocks(lambda x: np.unique(x)).compute())

    nclasses = len(unique_classes)

    gpu_futures = client.sync(_extract_partitions, [y_true, y_pred], client)

    cms = client.compute(
        [
            client.submit(
                local_cm, part, unique_classes, sample_weight, workers=[worker]
            )
            for worker, part in gpu_futures
        ],
        sync=True,
    )

    cm = np.zeros((nclasses, nclasses), like=y_true._meta)

    for i in cms:
        cm += i

    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()

        if isinstance(y_true._meta, cp.ndarray):
            cm = cp.nan_to_num(cm)
        else:
            cm = np.nan_to_num(cm)

    return cm


def accuracy_score(client, y, y_hat):

    gpu_futures = client.sync(_extract_partitions, [y_hat, y], client)

    def _count_accurate_predictions(y_hat_y):
        y_hat, y = y_hat_y

        if isinstance(y, cp.ndarray):
            return y.shape[0] - cp.count_nonzero(y - y_hat)
        else:
            return y.shape[0] - np.count_nonzero(y - y_hat)

    key = uuid1()

    futures = client.compute(
        [
            client.submit(
                _count_accurate_predictions,
                worker_future[1],
                workers=[worker_future[0]],
                key="%s-%s" % (key, idx),
            )
            for idx, worker_future in enumerate(gpu_futures)
        ],
        sync=True,
    )

    return sum(futures) / y.shape[0]    


def post_etl_processing(client, train_data, test_data):

    # Feature engineering
    X_train = build_features(train_data)
    X_test = build_features(test_data)

    y_train = build_labels(train_data)
    y_test = build_labels(test_data)

    # Perform ML
    if isinstance(y_train._meta, cp.ndarray):
        model = DistMNB(client=client, alpha=0.001)
        model.fit(X_train, y_train)
    else:
        model = ParallelPostFit(estimator=MultNB(alpha=0.001))
        model.fit(X_train.compute(), y_train.compute())

    ### this regression seems to be coming from here
    y_hat = model.predict(X_test).persist()

    # Compute distributed performance metrics
    acc = accuracy_score(client, y_test, y_hat)
    print("Accuracy: " + str(acc))

    prec = precision_score(client, y_test, y_hat, average="macro")
    print("Precision: " + str(prec))

    cmat = confusion_matrix(client, y_test, y_hat)
    print("Confusion Matrix: " + str(cmat))

    # Place results back in original Dataframe
    gpu_futures = client.sync(_extract_partitions, y_hat, client)
    
    ser_type = cudf.Series if isinstance(y_test._meta, cp.ndarray) else pd.Series

    test_preds = to_dask_cudf(
        [client.submit(ser_type, part) for w, part in gpu_futures]
    )

    test_preds = test_preds.map_partitions(categoricalize)

    test_data["prediction"] = test_preds

    final_data = test_data[["pr_review_sk", "pr_review_rating", "prediction"]].persist()

    final_data = final_data.sort_values("pr_review_sk").reset_index(drop=True)
    wait(final_data)
    return final_data, acc, prec, cmat


