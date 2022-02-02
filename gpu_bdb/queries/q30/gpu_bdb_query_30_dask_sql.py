#
<<<<<<< HEAD
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Copyright (c) 2019-2020, BlazingSQL, Inc.
=======
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
>>>>>>> 7b9e11852ec84bc74b965c24a911f5985123d951
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

<<<<<<< HEAD
import sys
import os

=======
>>>>>>> 7b9e11852ec84bc74b965c24a911f5985123d951
from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.sessionization import (
    get_distinct_sessions,
    get_pairs
)

<<<<<<< HEAD
from bdb_tools.readers import build_reader

from dask.distributed import wait


# -------- Q30 -----------
# session timeout in secs
q30_session_timeout_inSec = 3600
# query output limit
q30_limit = 40


def read_tables(data_dir, bc, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_cols = ["i_category_id", "i_item_sk"]
    item_df = table_reader.read("item", relevant_cols=item_cols)

    wcs_cols = ["wcs_user_sk", "wcs_item_sk", "wcs_click_date_sk", "wcs_click_time_sk"]
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)

    bc.create_table('web_clickstreams', wcs_df, persist=False)
    bc.create_table('item', item_df, persist=False)
    # print(len(wcs_df))

    # bc.create_table('web_clickstreams', os.path.join(data_dir, "web_clickstreams/*.parquet"))
    # bc.create_table('item', os.path.join(data_dir, "item/*.parquet"))


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, config, dask_profile=config["dask_profile"])
=======
from bdb_tools.q30_utils import (
    q30_session_timeout_inSec,
    q30_limit,
    read_tables
)

from dask.distributed import wait

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c, dask_profile=config["dask_profile"])
>>>>>>> 7b9e11852ec84bc74b965c24a911f5985123d951

    query_1 = """
        SELECT i_item_sk,
            CAST(i_category_id AS TINYINT) AS i_category_id
        FROM item
    """
<<<<<<< HEAD
    item_df = bc.sql(query_1)

    item_df = item_df.persist()
    wait(item_df)
    bc.create_table("item_df", item_df, persist=False)
    # print(len(item_df))
    # print(len(item_df.columns))
=======
    item_df = c.sql(query_1)

    item_df = item_df.persist()
    wait(item_df)
    c.create_table("item_df", item_df, persist=False)
>>>>>>> 7b9e11852ec84bc74b965c24a911f5985123d951

    query_2 = """
        SELECT wcs_user_sk,
            (wcs_click_date_sk * 86400 + wcs_click_time_sk) AS tstamp_inSec,
            i_category_id
        FROM web_clickstreams wcs, item_df i
        WHERE wcs.wcs_item_sk = i.i_item_sk
        AND i.i_category_id IS NOT NULL
        AND wcs.wcs_user_sk IS NOT NULL
<<<<<<< HEAD
        ORDER BY wcs.wcs_user_sk, tstamp_inSec, i_category_id
        DISTRIBUTE BY wcs_user_sk
    """
    merged_df = bc.sql(query_2)
    # print(len(merged_df))

    bc.drop_table("item_df")
=======
        DISTRIBUTE BY wcs_user_sk
    """
    merged_df = c.sql(query_2)

    c.drop_table("item_df")
>>>>>>> 7b9e11852ec84bc74b965c24a911f5985123d951
    del item_df

    distinct_session_df = merged_df.map_partitions(get_distinct_sessions,
            keep_cols=["wcs_user_sk", "i_category_id"],
            time_out=q30_session_timeout_inSec)

    del merged_df
    pair_df = distinct_session_df.map_partitions(
        get_pairs,
        pair_col="i_category_id",
        output_col_1="category_id_1",
        output_col_2="category_id_2")
    del distinct_session_df

<<<<<<< HEAD
    pair_df = pair_df.persist()
    wait(pair_df)
    bc.create_table('pair_df', pair_df, persist=False)
    # print(len(pair_df))
=======
    c.create_table('pair_df', pair_df, persist=False)
>>>>>>> 7b9e11852ec84bc74b965c24a911f5985123d951

    last_query = f"""
        SELECT CAST(category_id_1 AS BIGINT) AS category_id_1,
            CAST(category_id_2 AS BIGINT) AS category_id_2,
            COUNT(category_id_2) AS cnt
        FROM pair_df
        GROUP BY category_id_1, category_id_2
        ORDER BY cnt desc
        LIMIT {q30_limit}
    """
<<<<<<< HEAD
    result = bc.sql(last_query)

    bc.drop_table("pair_df")
=======
    result = c.sql(last_query)

    c.drop_table("pair_df")
>>>>>>> 7b9e11852ec84bc74b965c24a911f5985123d951
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
<<<<<<< HEAD
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
=======
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
>>>>>>> 7b9e11852ec84bc74b965c24a911f5985123d951
