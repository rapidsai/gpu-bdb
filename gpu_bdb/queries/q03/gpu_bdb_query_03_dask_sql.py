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

from bdb_tools.q03_utils import (
    apply_find_items_viewed
)

from dask.distributed import wait


# -------- Q03 -----------
q03_days_in_sec_before_purchase = 864000
q03_views_before_purchase = 5
q03_purchased_item_IN = 10001
# --see q1 for categories
q03_purchased_item_category_IN = "2,3"
q03_limit = 100


def read_tables(data_dir, c, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_cols = ["i_category_id", "i_item_sk"]
    wcs_cols = [
        "wcs_user_sk",
        "wcs_click_time_sk",
        "wcs_click_date_sk",
        "wcs_item_sk",
        "wcs_sales_sk",
    ]

    item_df = table_reader.read("item", relevant_cols=item_cols)
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)

    c.create_table("web_clickstreams", wcs_df, persist=False)
    c.create_table("item", item_df, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

    query_1 = """
        SELECT i_item_sk,
            CAST(i_category_id AS TINYINT) AS i_category_id
        FROM item
    """
    item_df = c.sql(query_1)

    item_df = item_df.persist()
    wait(item_df)
    c.create_table("item_df", item_df, persist=False)

    query_2 = """
        SELECT CAST(w.wcs_user_sk AS INTEGER) as wcs_user_sk,
            wcs_click_date_sk * 86400 + wcs_click_time_sk AS tstamp,
            CAST(w.wcs_item_sk AS INTEGER) as wcs_item_sk,
            CAST(COALESCE(w.wcs_sales_sk, 0) AS INTEGER) as wcs_sales_sk
        FROM web_clickstreams AS w
        INNER JOIN item_df AS i ON w.wcs_item_sk = i.i_item_sk
        WHERE w.wcs_user_sk IS NOT NULL
        AND w.wcs_item_sk IS NOT NULL
        DISTRIBUTE BY wcs_user_sk
    """
    merged_df = c.sql(query_2)

    query_3 = f"""
        SELECT i_item_sk, i_category_id
        FROM item_df
        WHERE i_category_id IN ({q03_purchased_item_category_IN})
    """
    item_df_filtered = c.sql(query_3)

    product_view_results = merged_df.map_partitions(
        apply_find_items_viewed, item_mappings=item_df_filtered
    )
    

    c.drop_table("item_df")
    del item_df
    del merged_df
    del item_df_filtered

    c.create_table('product_result', product_view_results, persist=False)

    last_query = f"""
        SELECT CAST({q03_purchased_item_IN} AS BIGINT) AS purchased_item,
            i_item_sk AS lastviewed_item,
            COUNT(i_item_sk) AS cnt
        FROM product_result
        GROUP BY i_item_sk
        ORDER BY purchased_item, cnt desc, lastviewed_item
        LIMIT {q03_limit}
    """
    result = c.sql(last_query)

    c.drop_table("product_result")
    del product_view_results
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
