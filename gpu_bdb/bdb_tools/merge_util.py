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

import dask
from dask.utils import apply
from dask.highlevelgraph import HighLevelGraph
from dask.dataframe.core import new_dd_object
from dask.dataframe.multi import merge_chunk


def hash_merge(
    lhs,
    left_on,
    rhs,
    right_on,
    how="inner",
    npartitions=None,
    suffixes=("_x", "_y"),
    shuffle=None,
    indicator=False,
):

    if npartitions is None:
        npartitions = max(lhs.npartitions, rhs.npartitions)

    lhs2 = lhs.shuffle(on=left_on, npartitions=npartitions)
    rhs2 = rhs.shuffle(on=right_on, npartitions=npartitions)

    kwargs = dict(
        how=how,
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
        indicator=indicator,
    )

    meta = lhs._meta_nonempty.merge(rhs._meta_nonempty, **kwargs)

    if isinstance(left_on, list):
        left_on = (list, tuple(left_on))

    if isinstance(right_on, list):
        right_on = (list, tuple(right_on))

    token = dask.base.tokenize(lhs2, rhs2, npartitions, shuffle, **kwargs)
    name = "hash-join-" + token

    kwargs["empty_index_dtype"] = meta.index.dtype
    dsk = {
        (name, i): (
            dask.utils.apply,
            merge_chunk,
            [(lhs2._name, i), (rhs2._name, i)],
            kwargs,
        )
        for i in range(npartitions)
    }

    divisions = [None] * (npartitions + 1)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[lhs2, rhs2])
    return new_dd_object(graph, name, meta, divisions)
