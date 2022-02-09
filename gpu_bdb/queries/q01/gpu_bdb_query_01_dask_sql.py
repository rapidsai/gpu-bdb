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

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q01_utils import (
    q01_i_category_id_IN,
    q01_ss_store_sk_IN,
    q01_viewed_together_count,
    q01_limit,
    read_tables
)

from dask.distributed import wait

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c)

    query_distinct = f"""
        SELECT DISTINCT ss_item_sk, ss_ticket_number
        FROM store_sales s, item i
        WHERE s.ss_item_sk = i.i_item_sk
        AND i.i_category_id IN {q01_i_category_id_IN}
        AND s.ss_store_sk IN {q01_ss_store_sk_IN}
    """
    result_distinct = c.sql(query_distinct)

    result_distinct = result_distinct.persist()
    wait(result_distinct)
    c.create_table("distinct_table", result_distinct, persist=False)

    query = f"""
        SELECT item_sk_1, item_sk_2, COUNT(*) AS cnt
        FROM
        (
            SELECT CAST(t1.ss_item_sk as BIGINT) AS item_sk_1,
                CAST(t2.ss_item_sk AS BIGINT) AS item_sk_2
            FROM distinct_table t1
            INNER JOIN distinct_table t2
            ON t1.ss_ticket_number = t2.ss_ticket_number
            WHERE t1.ss_item_sk < t2.ss_item_sk
        )
        GROUP BY item_sk_1, item_sk_2
        HAVING  COUNT(*) > {q01_viewed_together_count}
        ORDER BY cnt DESC, CAST(item_sk_1 AS VARCHAR),
                 CAST(item_sk_2 AS VARCHAR)
        LIMIT {q01_limit}
    """
    result = c.sql(query)

    c.drop_table("distinct_table")
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
