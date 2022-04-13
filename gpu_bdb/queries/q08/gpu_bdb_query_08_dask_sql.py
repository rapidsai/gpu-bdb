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

import dask_cudf
import pandas as pd
import dask.dataframe as dd

from nvtx import annotate
from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q08_utils import (
    get_sessions,
    get_unique_sales_keys_from_sessions,
    prep_for_sessionization,
    q08_STARTDATE,
    q08_ENDDATE,
    read_tables
)

from dask.distributed import wait

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c)

    query_1 = f"""
        SELECT d_date_sk
        FROM date_dim
        WHERE CAST(d_date as date) IN (date '{q08_STARTDATE}',
                                       date '{q08_ENDDATE}')
        ORDER BY CAST(d_date as date) asc
    """
    result_dates_sk_filter = c.sql(query_1).compute()

    # because `result_dates_sk_filter` has repetitive index
    result_dates_sk_filter.index = list(range(0, result_dates_sk_filter.shape[0]))
    q08_start_dt = result_dates_sk_filter['d_date_sk'][0]
    q08_end_dt = result_dates_sk_filter['d_date_sk'][1]

    query_aux = """
        SELECT
            wp_web_page_sk,
            wp_type
        FROM web_page
    """
    web_page_df = c.sql(query_aux)

    # cast to minimum viable dtype
    web_page_df["wp_type"] = web_page_df["wp_type"].map_partitions(
        lambda ser: ser.astype("category")
    )

    cpu_categories = web_page_df["wp_type"].compute().cat.categories
        
    if isinstance(web_page_df, dask_cudf.DataFrame):    
        cpu_categories = cpu_categories.to_pandas()
        
    REVIEW_CAT_CODE = cpu_categories.get_loc("review")

    web_page_df["wp_type_codes"] = web_page_df["wp_type"].cat.codes

    web_page_newcols = ["wp_web_page_sk", "wp_type_codes"]
    web_page_df = web_page_df[web_page_newcols]

    web_page_df = web_page_df.persist()
    wait(web_page_df)
    c.create_table('web_page_2', web_page_df, persist=False)

    query_2 = f"""
        SELECT
            CAST(wcs_user_sk AS INTEGER) AS wcs_user_sk,
            (wcs_click_date_sk * 86400 + wcs_click_time_sk) AS tstamp_inSec,
            wcs_sales_sk,
            wp_type_codes
        FROM web_clickstreams
        INNER JOIN web_page_2 ON wcs_web_page_sk = wp_web_page_sk
        WHERE wcs_user_sk IS NOT NULL
        AND wcs_click_date_sk BETWEEN {q08_start_dt} AND {q08_end_dt}
        --in the future we want to remove this ORDER BY
        DISTRIBUTE BY wcs_user_sk
    """
    merged_df = c.sql(query_2)

    c.drop_table("web_page_2")
    del web_page_df
    
    merged_df = merged_df.shuffle(on=["wcs_user_sk"])
    merged_df["review_flag"] = merged_df.wp_type_codes == REVIEW_CAT_CODE

    prepped = merged_df.map_partitions(
        prep_for_sessionization, review_cat_code=REVIEW_CAT_CODE
    )

    sessionized = prepped.map_partitions(get_sessions)

    unique_review_sales = sessionized.map_partitions(
        get_unique_sales_keys_from_sessions, review_cat_code=REVIEW_CAT_CODE
    )
    if isinstance(merged_df, dask_cudf.DataFrame):
        unique_review_sales = unique_review_sales.to_frame()
    else: 
        unique_review_sales = dd.from_dask_array(unique_review_sales, columns='wcs_sales_sk').to_frame() 
        
    unique_review_sales = unique_review_sales.persist()
    wait(unique_review_sales)
    c.create_table("reviews", unique_review_sales, persist=False)
    last_query = f"""
        SELECT
            CAST(review_total AS BIGINT) AS q08_review_sales_amount,
            CAST(total - review_total AS BIGINT) AS no_q08_review_sales_amount
        FROM
        (
            SELECT
            SUM(ws_net_paid) AS total,
            SUM(CASE when wcs_sales_sk IS NULL THEN 0 ELSE 1 END * ws_net_paid) AS review_total
            FROM web_sales
            LEFT OUTER JOIN reviews ON ws_order_number = wcs_sales_sk
            WHERE ws_sold_date_sk between {q08_start_dt} AND {q08_end_dt}
        )
    """
    result = c.sql(last_query)

    c.drop_table("reviews")
    return result


@annotate("QUERY8", color="green", domain="gpu-bdb")
def start_run():
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)

if __name__ == "__main__":
    start_run()    
