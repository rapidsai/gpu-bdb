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

from bdb_tools.readers import build_reader

from bdb_tools.q28_utils import (
    post_etl_processing,
    read_tables
)

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c, dask_profile=config["dask_profile"])

    # 10 % of data
    query1 = """
        SELECT
            pr_review_sk,
            pr_review_rating,
            pr_review_content
        FROM product_reviews
        WHERE mod(pr_review_sk, 10) IN (0)
        AND pr_review_content IS NOT NULL
        ORDER BY pr_review_sk
    """
    test_data = c.sql(query1)

    # 90 % of data
    query2 = """
        SELECT
            pr_review_sk,
            pr_review_rating,
            pr_review_content
        FROM product_reviews
        WHERE mod(pr_review_sk, 10) IN (1,2,3,4,5,6,7,8,9)
        AND pr_review_content IS NOT NULL
        ORDER BY pr_review_sk
    """
    train_data = c.sql(query2)

    final_data, acc, prec, cmat = post_etl_processing(
        client=client, train_data=train_data, test_data=test_data
    )

    payload = {
        "df": final_data,
        "acc": acc,
        "prec": prec,
        "cmat": cmat,
        "output_type": "supervised",
    }

    return payload


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
