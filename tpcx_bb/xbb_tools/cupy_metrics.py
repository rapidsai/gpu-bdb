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

"""
Implement simple scoring metrics using cupy (instead of going via host)

Should be removed once cuML impelements their own scoring metrics.

Thanks @beckernick for providing the initial implementations.
"""
import cupy as cp


def cupy_conf_mat(y, y_pred):
    """
    Simple, fast confusion matrix for two class models designed to match sklearn.
    Assumes the classes are one of [0, 1]. It will fail edge cases, which are fairly
    numerous.

    Implementation taken from rapidsai/cuml#1524
    """
    nclasses = len(cp.unique(y))
    assert nclasses == 2
    res = cp.zeros((2, 2))

    pos_pred_ix = cp.where(y_pred == 1)
    neg_pred_ix = cp.where(y_pred != 1)
    tn_sum = (y[neg_pred_ix] == 0).sum()
    fn_sum = (y[neg_pred_ix] == 1).sum()
    tp_sum = (y[pos_pred_ix] == 1).sum()
    fp_sum = (y[pos_pred_ix] == 0).sum()

    res[0, 0] = tn_sum
    res[1, 0] = fn_sum
    res[0, 1] = fp_sum
    res[1, 1] = tp_sum
    return res


def cupy_precision_score(y, y_pred):
    """
    Simple, precision score method for two class models.
    It is assumed that the positive class has label 1.

    Implementstion taken from rapidsai/cuml#1522
    """
    pos_pred_ix = cp.where(y_pred == 1)
    tp_sum = (y_pred[pos_pred_ix] == y[pos_pred_ix]).sum()
    fp_sum = (y_pred[pos_pred_ix] != y[pos_pred_ix]).sum()

    return (tp_sum / (tp_sum + fp_sum)).item()
