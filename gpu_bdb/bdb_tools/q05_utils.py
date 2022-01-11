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

import cupy as cp

import cuml
from cuml.metrics import confusion_matrix

from bdb_tools.cupy_metrics import cupy_precision_score

from sklearn.metrics import roc_auc_score

# Logistic Regression params
# solver = "LBFGS" Used by passing `penalty=None` or "l2"
# step_size = 1 Not used
# numCorrections = 10 Not used
iterations = 100
C = 10_000  # reg_lambda = 0 hence C for model is a large value
convergence_tol = 1e-9


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

