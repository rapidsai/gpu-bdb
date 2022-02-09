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

from bdb_tools.q15_utils import (
    q15_startDate,
    q15_endDate,
    q15_store_sk,
    read_tables
)

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c)

    query = f"""
        SELECT *
        FROM
        (
            SELECT
                cat,
                ( (count(x) * SUM(xy) - SUM(x) * SUM(y)) / (count(x) * SUM(xx) - SUM(x) * SUM(x)) )  AS slope,
                (SUM(y) - ((count(x) * SUM(xy) - SUM(x) * SUM(y)) / (count(x) * SUM(xx) - SUM(x)*SUM(x)) ) * SUM(x)) / count(x) AS intercept
            FROM
            (
                SELECT
                    i.i_category_id AS cat,
                    s.ss_sold_date_sk AS x,
                    CAST(SUM(s.ss_net_paid) AS DOUBLE) AS y,
                    CAST(s.ss_sold_date_sk * SUM(s.ss_net_paid) AS DOUBLE) AS xy,
                    CAST(s.ss_sold_date_sk * s.ss_sold_date_sk AS DOUBLE) AS xx
                FROM store_sales s
                INNER JOIN item i ON s.ss_item_sk = i.i_item_sk
                INNER JOIN date_dim d ON s.ss_sold_date_sk = d.d_date_sk
                WHERE s.ss_store_sk = {q15_store_sk}
                AND i.i_category_id IS NOT NULL
                AND CAST(d.d_date AS DATE) >= DATE '{q15_startDate}'
                AND   CAST(d.d_date AS DATE) <= DATE '{q15_endDate}'
                GROUP BY i.i_category_id, s.ss_sold_date_sk
            ) temp
            GROUP BY cat
        ) regression
        WHERE slope <= 0.0
        ORDER BY cat
    """
    result = c.sql(query)
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
