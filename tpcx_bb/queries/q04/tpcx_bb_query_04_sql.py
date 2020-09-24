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

from xbb_tools.cluster_startup import attach_to_cluster
from xbb_tools.sessionization import get_sessions

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

from dask.distributed import wait

def abandonedShoppingCarts(df, DYNAMIC_CAT_CODE, ORDER_CAT_CODE):
    import cudf
    # work around for https://github.com/rapidsai/cudf/issues/5470
    df.reset_index(drop=True, inplace=True)

    # Select groups where last dynamic row comes after last order row
    filtered_df = df[
        (df["wp_type_codes"] == ORDER_CAT_CODE)
        | (df["wp_type_codes"] == DYNAMIC_CAT_CODE)
    ]
    # work around for https://github.com/rapidsai/cudf/issues/5470
    filtered_df.reset_index(drop=True, inplace=True)
    # Create a new column that is the concatenation of timestamp and wp_type_codes
    # (eg:123456:3, 234567:5)
    filtered_df["wp_type_codes"] = (
        filtered_df["tstamp_inSec"]
        .astype("str")
        .str.cat(filtered_df["wp_type_codes"].astype("str"), sep=":")
    )
    # This gives the last occurrence (by timestamp) within the "order", "dynamic" wp_types
    filtered_df = filtered_df.groupby(
        ["wcs_user_sk", "session_id"], as_index=False, sort=False
    ).agg({"wp_type_codes": "max"})
    # If the max contains dynamic, keep the row else discard.
    last_dynamic_df = filtered_df[
        filtered_df["wp_type_codes"].str.contains(
            ":" + str(DYNAMIC_CAT_CODE), regex=False
        )
    ]
    del filtered_df

    # Find counts for each group
    grouped_count_df = df.groupby(
        ["wcs_user_sk", "session_id"], as_index=False, sort=False
    ).agg({"tstamp_inSec": "count"})
    # Merge counts with the "dynamic" shopping cart groups
    result = last_dynamic_df.merge(
        grouped_count_df, on=["wcs_user_sk", "session_id"], how="inner"
    )
    del (last_dynamic_df, grouped_count_df)
    return cudf.DataFrame(
        {"pagecount": result.tstamp_inSec.sum(), "count": len(result)}
    )


def reduction_function(df, keep_cols, DYNAMIC_CAT_CODE, ORDER_CAT_CODE):
    df = get_sessions(df, keep_cols=keep_cols)
    df = abandonedShoppingCarts(
        df, DYNAMIC_CAT_CODE=DYNAMIC_CAT_CODE, ORDER_CAT_CODE=ORDER_CAT_CODE
    )
    return df


def read_tables(data_dir, bc):
    bc.create_table('web_page_wo_categorical', data_dir + "web_page/*.parquet")
    bc.create_table('web_clickstreams',
                    data_dir + "web_clickstreams/*.parquet")


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query_web_page = """
        SELECT wp_type, wp_web_page_sk
        FROM web_page_wo_categorical
    """
    wp = bc.sql(query_web_page)

    # Convert wp_type to categorical and get cat_id of review and dynamic type
    wp["wp_type"] = wp["wp_type"].map_partitions(
                                    lambda ser: ser.astype("category"))
    cpu_categories = wp["wp_type"].compute().cat.categories.to_pandas()
    DYNAMIC_CAT_CODE = cpu_categories.get_loc("dynamic")
    ORDER_CAT_CODE = cpu_categories.get_loc("order")

    # ### cast to minimum viable dtype
    import cudf
    codes_min_signed_type = cudf.utils.dtypes.min_signed_type(
                                                    len(cpu_categories))
    wp["wp_type_codes"] = wp["wp_type"].cat.codes.astype(codes_min_signed_type)
    wp["wp_type"] = wp["wp_type"].cat.codes.astype(codes_min_signed_type)
    cols_2_keep = ["wp_web_page_sk", "wp_type_codes"]
    wp = wp[cols_2_keep]

    wp = wp.persist()
    wait(wp)
    bc.create_table('web_page', wp)

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
        ORDER BY wcs_user_sk, tstamp_inSec
    """
    merged_df = bc.sql(query)

    keep_cols = ["wcs_user_sk", "wp_type_codes", "tstamp_inSec"]
    result_df = merged_df.map_partitions(
        reduction_function, keep_cols, DYNAMIC_CAT_CODE, ORDER_CAT_CODE
    )

    result = result_df["pagecount"].sum() / result_df["count"].sum()
    # Persist before computing to ensure scalar transfer only on compute
    result = result.persist()

    result = result.compute()
    result_df = cudf.DataFrame({"sum(pagecount)/count(*)": [result]})
    bc.drop_table("web_page")
    return result_df


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
