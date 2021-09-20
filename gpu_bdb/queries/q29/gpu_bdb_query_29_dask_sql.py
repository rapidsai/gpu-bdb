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
import os

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader

from dask.distributed import wait

from dask_sql import Context

# -------- Q29 -----------
q29_limit = 100


def read_tables(data_dir, bc):
    table_reader = build_reader(
        data_format=config["file_format"], basepath=config["data_dir"],
    )
    item_cols = ["i_item_sk", "i_category_id"]
    item_df = table_reader.read("item", relevant_cols=item_cols)

    ws_cols = ["ws_order_number", "ws_item_sk"]
    ws_df = table_reader.read("web_sales", relevant_cols=ws_cols)

    bc.create_table('item', item_df, persist=False)
    bc.create_table('web_sales', ws_df, persist=False)


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])
    n_workers = len(client.scheduler_info()["workers"])

    join_query = """
        -- Commented Distinct as we do it in
        -- dask_cudf based drop_duplicates with drop_duplicates
        -- 553 M rows dont fit on single GPU (int32,int64 column)
        -- TODO: Remove when we support Split Out
        -- https://github.com/dask-contrib/dask-sql/issues/241

        SELECT  i_category_id, ws_order_number
        FROM web_sales ws, item i
        WHERE ws.ws_item_sk = i.i_item_sk
        AND i.i_category_id IS NOT NULL
    """
    result = bc.sql(join_query)
    
    # Distinct Calculatiin
    result_distinct = result.drop_duplicates(split_out=n_workers,ignore_index=True)
    ## Remove the int64 index that was created
    ## TODO Raise a issue for this
    result_distinct = result_distinct.reset_index(drop=True)
    ### Persiting cause Order by causes execution
    bc.create_table('distinct_table', result_distinct, persist=True)

    query = f"""
        SELECT category_id_1, category_id_2, COUNT (*) AS cnt
        FROM
        (
            SELECT CAST(t1.i_category_id as BIGINT) AS category_id_1,
                CAST(t2.i_category_id as BIGINT) AS category_id_2
            FROM distinct_table t1
            INNER JOIN distinct_table t2
            ON t1.ws_order_number = t2.ws_order_number
            WHERE t1.i_category_id < t2.i_category_id
        )
        GROUP BY category_id_1, category_id_2
        ORDER BY cnt DESC, category_id_1, category_id_2
        LIMIT {q29_limit}
    """
    result = bc.sql(query)
    result = result.persist()
    wait(result);

    bc.drop_table("distinct_table")
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, _ = attach_to_cluster(config)
    c = Context()
    run_query(config=config, client=client, query_func=main, blazing_context=c)
