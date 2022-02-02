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

import cudf

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q04_utils import (
    reduction_function,
    read_tables
)

from dask.distributed import wait


def main(data_dir, client, c, config):
    benchmark(read_tables, config, c, dask_profile=config["dask_profile"])

    query_web_page = """
        SELECT wp_type, wp_web_page_sk
        FROM web_page_wo_categorical
    """
    wp = c.sql(query_web_page)

    # Convert wp_type to categorical and get cat_id of review and dynamic type
    wp["wp_type"] = wp["wp_type"].map_partitions(
                                    lambda ser: ser.astype("category"))
    
    cpu_categories = wp["wp_type"].compute().cat.categories.to_pandas()

    DYNAMIC_CAT_CODE = cpu_categories.get_loc("dynamic")
    ORDER_CAT_CODE = cpu_categories.get_loc("order")

    # ### cast to minimum viable dtype
    wp["wp_type_codes"] = wp["wp_type"].cat.codes
    cols_2_keep = ["wp_web_page_sk", "wp_type_codes"]
    wp = wp[cols_2_keep]

    wp = wp.persist()
    wait(wp)
    c.create_table('web_page', wp, persist=False)

    query = """
        SELECT
            c.wcs_user_sk,
            w.wp_type_codes,
            (wcs_click_date_sk * 86400 + wcs_click_time_sk) AS tstamp_inSec
        FROM web_clickstreams c, web_page w
        WHERE c.wcs_web_page_sk = w.wp_web_page_sk
        AND   c.wcs_web_page_sk IS NOT NULL
        AND   c.wcs_user_sk     IS NOT NULL
        AND   c.wcs_sales_sk    IS NULL --abandoned implies: no sale
        DISTRIBUTE BY wcs_user_sk
    """
    merged_df = c.sql(query)

    keep_cols = ["wcs_user_sk", "wp_type_codes", "tstamp_inSec"]
    result_df = merged_df.map_partitions(
        reduction_function, keep_cols, DYNAMIC_CAT_CODE, ORDER_CAT_CODE
    )

    result = result_df["pagecount"].sum() / result_df["count"].sum()
    # Persist before computing to ensure scalar transfer only on compute
    result = result.persist()

    result = result.compute()
    result_df = cudf.DataFrame({"sum(pagecount)/count(*)": [result]})
    c.drop_table("web_page")
    return result_df


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
