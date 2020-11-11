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

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)
from xbb_tools.readers import build_reader
from xbb_tools.utils import benchmark
from distributed import wait

### Implementation Notes:
# * `drop_duplicates` and `groupby` by default brings result to single partition
#    * Have changed `drop_duplicates` behaviour to give `n_workers` partitions
#    * This can change performence chars at larger scales

### Future Notes:
# Setting  index + merge using  map_parition can be a work-around if dask native merge is slow
#    * Note:Set index isssue: https://github.com/rapidsai/cudf/issues/2272
#    * Implientation Idea:
#        * After setting index, We can do a (inner join + group-by) with better parallelism and less data movement

### Scalabilty problems
# * The ws_item_join table after distincts has `48M` rows, can cause problems on bigger scale factors


# -------- Q29 -----------
q29_limit = 100
q29_session_timeout_inSec = 3600


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"], basepath=config["data_dir"],
    )
    item_cols = ["i_item_sk", "i_category_id"]
    item_df = table_reader.read("item", relevant_cols=item_cols)

    ws_cols = ["ws_order_number", "ws_item_sk"]
    ws_df = table_reader.read("web_sales", relevant_cols=ws_cols)

    return item_df, ws_df


###
# Select t1.i_category_id AS category_id_1 , t2.i_category_id AS category_id_2
#  FROM (
#    ...
# ) t1 Inner Join
#    (
#    ...
#    ) t2
#  ON t1.ws_order_number == t2.ws_order_number
#  WHERE
#  t1.i_category_id<t2.i_category_id
#  )
###


def get_pairs(
    df,
    merge_col="ws_order_number",
    pair_col="i_category_id",
    output_col_1="category_id_1",
    output_col_2="category_id_2",
):
    """
        Gets pair after doing a inner merge
    """
    pair_df = df.merge(df, on=merge_col, suffixes=["_t1", "_t2"], how="inner")
    pair_df = pair_df[[f"{pair_col}_t1", f"{pair_col}_t2"]]
    pair_df = pair_df[pair_df[f"{pair_col}_t1"] < pair_df[f"{pair_col}_t2"]]
    pair_df = pair_df.rename(
        columns={f"{pair_col}_t1": output_col_1, f"{pair_col}_t2": output_col_2}
    )
    return pair_df


def main(client, config):

    item_df, ws_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )
    ### setting index on ws_order_number
    ws_df = ws_df.shuffle(on=["ws_order_number"])
    ### at sf-100k we will have max of 17M rows and 17 M rows with 2 columns, 1 part is very reasonable
    item_df = item_df.repartition(npartitions=1)

    # SELECT DISTINCT i_category_id, ws_order_number
    # FROM web_sales ws, item i
    # WHERE ws.ws_item_sk = i.i_item_sk
    # AND i.i_category_id IS NOT NULL

    f_item_df = item_df[item_df["i_category_id"].notnull()]
    ### doing below to retain the `ws_order_number` partition boundry after merge
    ws_item_join = ws_df.merge(
        f_item_df, left_on=["ws_item_sk"], right_on=["i_item_sk"]
    )
    ws_item_join = ws_item_join[["i_category_id", "ws_order_number"]]
    ws_item_join = ws_item_join.map_partitions(lambda df: df.drop_duplicates())

    ### do pair inner join
    ### pair_df =  get_pairs(ws_item_join)
    ### because of setting index we can do it in map_partitions
    ### this can have  better memory and scaling props at larger scale factors
    pair_df = ws_item_join.map_partitions(get_pairs)

    # SELECT category_id_1, category_id_2, COUNT (*) AS cnt
    # FROM (
    #  ...
    #    )
    # GROUP BY category_id_1, category_id_2
    # ORDER BY cnt DESC, category_id_1, category_id_2
    # LIMIT {q29_limit}
    grouped_df = pair_df.groupby(["category_id_1", "category_id_2"]).size().persist()

    ### 36 rows after filtration at sf-100
    ### should scale till sf-100k
    grouped_df = grouped_df.reset_index().compute()
    grouped_df.columns = ["category_id_1", "category_id_2", "cnt"]
    grouped_df["category_id_1"] = grouped_df["category_id_1"]
    grouped_df["category_id_2"] = grouped_df["category_id_2"]
    grouped_df = grouped_df.sort_values(
        by=["cnt", "category_id_1", "category_id_2"], ascending=[False, True, True]
    ).reset_index(drop=True)
    grouped_df = grouped_df.head(q29_limit)

    return grouped_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
