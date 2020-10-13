#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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


from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)
from xbb_tools.readers import build_reader
from xbb_tools.sessionization import get_sessions


# parameters
q04_session_timeout_inSec = 3600


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    wp_cols = ["wp_type", "wp_web_page_sk"]
    wp_df = table_reader.read("web_page", relevant_cols=wp_cols)

    wcs_cols = [
        "wcs_user_sk",
        "wcs_click_date_sk",
        "wcs_click_time_sk",
        "wcs_web_page_sk",
        "wcs_sales_sk",
    ]
    web_clicksteams_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)

    return wp_df, web_clicksteams_df


def abandonedShoppingCarts(df, DYNAMIC_CAT_CODE, ORDER_CAT_CODE):
    import cudf

    # TODO: test without reset index
    df.reset_index(drop=True, inplace=True)

    # Select groups where last dynamic row comes after last order row
    filtered_df = df[
        (df["wp_type_codes"] == ORDER_CAT_CODE)
        | (df["wp_type_codes"] == DYNAMIC_CAT_CODE)
    ]
    # TODO: test without reset index
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


def main(client, config):
    import cudf

    wp, wcs_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    ### downcasting the column inline with q03
    wcs_df["wcs_user_sk"] = wcs_df["wcs_user_sk"].astype("int32")

    f_wcs_df = wcs_df[
        wcs_df["wcs_web_page_sk"].notnull()
        & wcs_df["wcs_user_sk"].notnull()
        & wcs_df["wcs_sales_sk"].isnull()
    ].reset_index(drop=True)

    f_wcs_df["tstamp_inSec"] = (
        f_wcs_df["wcs_click_date_sk"] * 24 * 60 * 60 + f_wcs_df["wcs_click_time_sk"]
    )
    keep_cols = ["wcs_user_sk", "tstamp_inSec", "wcs_web_page_sk"]
    f_wcs_df = f_wcs_df[keep_cols]

    f_wcs_df = f_wcs_df.shuffle(on=["wcs_user_sk"])

    # Convert wp_type to categorical and get cat_id of review and dynamic type
    wp["wp_type"] = wp["wp_type"].map_partitions(lambda ser: ser.astype("category"))
    cpu_categories = wp["wp_type"].compute().cat.categories.to_pandas()
    DYNAMIC_CAT_CODE = cpu_categories.get_loc("dynamic")
    ORDER_CAT_CODE = cpu_categories.get_loc("order")
    # ### cast to minimum viable dtype
    codes_min_signed_type = cudf.utils.dtypes.min_signed_type(len(cpu_categories))
    wp["wp_type_codes"] = wp["wp_type"].cat.codes.astype(codes_min_signed_type)
    cols_2_keep = ["wp_web_page_sk", "wp_type_codes"]
    wp = wp[cols_2_keep]

    # Continue remaining workflow with wp_type as category codes
    merged_df = f_wcs_df.merge(
        wp, left_on="wcs_web_page_sk", right_on="wp_web_page_sk", how="inner"
    )
    merged_df = merged_df[["wcs_user_sk", "tstamp_inSec", "wp_type_codes"]]

    keep_cols = ["wcs_user_sk", "wp_type_codes", "tstamp_inSec"]
    result_df = merged_df.map_partitions(
        reduction_function, keep_cols, DYNAMIC_CAT_CODE, ORDER_CAT_CODE
    )

    result = result_df["pagecount"].sum() / result_df["count"].sum()
    # Persist before computing to ensure scalar transfer only on compute
    result = result.persist()

    result = result.compute()
    result_df = cudf.DataFrame({"sum(pagecount)/count(*)": [result]})
    return result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
