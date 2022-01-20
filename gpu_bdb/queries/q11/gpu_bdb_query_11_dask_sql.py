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
import cudf

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q11_utils import read_tables

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c, dask_profile=config["dask_profile"])

    query = """
        WITH p AS
        (
            SELECT
                pr_item_sk,
                count(pr_item_sk) AS r_count,
                AVG( CAST(pr_review_rating AS DOUBLE) ) avg_rating  
            FROM product_reviews
            WHERE pr_item_sk IS NOT NULL
            GROUP BY pr_item_sk
        ), s AS
        (
            SELECT
                ws_item_sk
            FROM web_sales ws
            INNER JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
            WHERE ws_item_sk IS NOT null
            AND CAST(d.d_date AS DATE) >= DATE '2003-01-02'
            AND CAST(d.d_date AS DATE) <= DATE '2003-02-02'
            GROUP BY ws_item_sk
        )
        SELECT p.r_count    AS x,
            p.avg_rating AS y
        FROM s INNER JOIN p ON p.pr_item_sk = s.ws_item_sk
    """

    result = c.sql(query)
    sales_corr = result["x"].corr(result["y"]).compute()
    result_df = cudf.DataFrame([sales_corr])
    result_df.columns = ["corr(CAST(reviews_count AS DOUBLE), avg_rating)"]
    return result_df


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
