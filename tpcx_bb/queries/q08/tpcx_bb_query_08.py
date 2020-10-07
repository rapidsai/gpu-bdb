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


import glob

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
    convert_datestring_to_days,
)
from xbb_tools.readers import build_reader
from xbb_tools.merge_util import hash_merge

import numpy as np
from distributed import wait
import cupy as cp
import rmm
from dask import delayed


q08_STARTDATE = "2001-09-02"
q08_ENDDATE = "2002-09-02"
q08_SECONDS_BEFORE_PURCHASE = 259200
NA_FLAG = 0


def etl_wcs(wcs_fn, filtered_date_df, web_page_df):
    import cudf

    filtered_date_df = filtered_date_df
    web_page_df = web_page_df

    web_clickstreams_cols = [
        "wcs_user_sk",
        "wcs_click_date_sk",
        "wcs_sales_sk",
        "wcs_web_page_sk",
        "wcs_click_time_sk",
    ]
    web_clickstreams_df = cudf.read_parquet(wcs_fn, columns=web_clickstreams_cols)

    web_clickstreams_df = web_clickstreams_df[
        web_clickstreams_df["wcs_user_sk"].notnull()
    ].reset_index(drop=True)

    merged_df = web_clickstreams_df.merge(
        filtered_date_df,
        right_on=["d_date_sk"],
        left_on=["wcs_click_date_sk"],
        how="inner",
    )

    merged_df = merged_df.merge(
        web_page_df,
        left_on=["wcs_web_page_sk"],
        right_on=["wp_web_page_sk"],
        how="inner",
    )

    ### decrease column after merge
    merged_df["tstamp_inSec"] = (
        merged_df["wcs_click_date_sk"] * 86400 + merged_df["wcs_click_time_sk"]
    )
    cols_to_keep = ["wcs_user_sk", "tstamp_inSec", "wcs_sales_sk", "wp_type_codes"]
    return merged_df[cols_to_keep]


def get_session_id_from_session_boundary(session_change_df, last_session_len):
    """
        This function returns session starts given a session change df
    """
    import cudf

    user_session_ids = session_change_df.tstamp_inSec

    ### up shift the session length df
    session_len = session_change_df["t_index"].diff().reset_index(drop=True)
    session_len = session_len.shift(-1)

    try:
        session_len.iloc[-1] = last_session_len
    except (AssertionError, IndexError) as e:  # IndexError in numba >= 0.48
        session_len = cudf.Series([])

    session_id_final_series = (
        cudf.Series(user_session_ids).repeat(session_len).reset_index(drop=True)
    )
    return session_id_final_series


def get_session_id(df):
    """
        This function creates a session id column for each click
        The session id grows in incremeant for each user's susbequent session
        Session boundry is defined by the time_out
    """

    df["user_change_flag"] = df["wcs_user_sk"].diff(periods=1) != 0
    df["session_change_flag"] = df["review_flag"] | df["user_change_flag"]

    df = df.reset_index(drop=True)
    df["t_index"] = cp.arange(start=0, stop=len(df), dtype=np.int32)

    session_change_df = df[df["session_change_flag"]].reset_index(drop=True)
    try:
        last_session_len = len(df) - session_change_df["t_index"].iloc[-1]
    except (AssertionError, IndexError) as e:  # IndexError in numba >= 0.48
        last_session_len = 0

    session_ids = get_session_id_from_session_boundary(
        session_change_df, last_session_len
    )

    assert len(session_ids) == len(df)
    return session_ids


def get_sessions(df):
    df = df.sort_values(
        by=["wcs_user_sk", "tstamp_inSec", "wcs_sales_sk", "wp_type_codes"]
    ).reset_index(drop=True)
    df["session_id"] = get_session_id(df)
    return df


def get_unique_sales_keys_from_sessions(sessionized, review_cat_code):
    sessionized["relevant"] = (
        (sessionized.tstamp_inSec - sessionized.session_id)
        <= q08_SECONDS_BEFORE_PURCHASE
    ) & (sessionized.wcs_sales_sk != NA_FLAG)
    unique_sales_sk = (
        sessionized.query(f"wcs_sales_sk != {NA_FLAG}")
        .query("relevant == True")
        .query(f"wp_type_codes != {review_cat_code}")
        .wcs_sales_sk.unique()
    )

    return unique_sales_sk


def prep_for_sessionization(df, review_cat_code):
    df = df.fillna(NA_FLAG)
    df = df.sort_values(
        by=["wcs_user_sk", "tstamp_inSec", "wcs_sales_sk", "wp_type_codes"]
    ).reset_index(drop=True)

    review_df = df.loc[df["wp_type_codes"] == review_cat_code]
    # per user, the index of the first review
    # need this to decide if a review was "recent enough"
    every_users_first_review = (
        review_df[["wcs_user_sk", "tstamp_inSec"]]
        .drop_duplicates()
        .reset_index()
        .groupby("wcs_user_sk")["index"]
        .min()
        .reset_index()
    )
    every_users_first_review.columns = ["wcs_user_sk", "first_review_index"]

    # then reset the index to keep the old index before parallel join
    df_merged = df.reset_index().merge(
        every_users_first_review, how="left", on="wcs_user_sk"
    )
    df_filtered = df_merged.query("index >= first_review_index")
    return df_filtered


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    date_dim_cols = ["d_date_sk", "d_date"]
    web_page_cols = ["wp_web_page_sk", "wp_type"]
    web_sales_cols = ["ws_net_paid", "ws_order_number", "ws_sold_date_sk"]

    date_dim_df = table_reader.read("date_dim", relevant_cols=date_dim_cols)
    web_page_df = table_reader.read("web_page", relevant_cols=web_page_cols)
    web_sales_df = table_reader.read("web_sales", relevant_cols=web_sales_cols)

    return (date_dim_df, web_page_df, web_sales_df)


def reduction_function(df, REVIEW_CAT_CODE):

    # category code of review records
    df["review_flag"] = df.wp_type_codes == REVIEW_CAT_CODE

    # set_index in the previous statement will make sure all records of each wcs_user_sk end up in one partition.
    df = prep_for_sessionization(df, review_cat_code=REVIEW_CAT_CODE)
    df = get_sessions(df)
    df = get_unique_sales_keys_from_sessions(df, REVIEW_CAT_CODE)
    return df.to_frame()


def main(client, config):
    import cudf
    import dask_cudf

    (date_dim_df, web_page_df, web_sales_df) = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    date_dim_cov_df = date_dim_df.map_partitions(convert_datestring_to_days)
    q08_start_dt = np.datetime64(q08_STARTDATE, "D").astype(int)
    q08_end_dt = np.datetime64(q08_ENDDATE, "D").astype(int)
    filtered_date_df = date_dim_cov_df.query(
        f"d_date >= {q08_start_dt} and d_date <= {q08_end_dt}",
        meta=date_dim_cov_df._meta,
    ).reset_index(drop=True)

    # Convert wp_type to categorical and get cat_id of review and dynamic type
    # see https://github.com/rapidsai/cudf/issues/4093 for more info
    web_page_df = web_page_df.persist()

    # map_partitions is a bit faster than ddf[col].astype('category')
    web_page_df["wp_type"] = web_page_df["wp_type"].map_partitions(
        lambda ser: ser.astype("category")
    )
    cpu_categories = web_page_df["wp_type"].compute().cat.categories.to_pandas()
    REVIEW_CAT_CODE = cpu_categories.get_loc("review")

    # cast to minimum viable dtype
    codes_min_signed_type = cudf.utils.dtypes.min_signed_type(len(cpu_categories))

    web_page_df["wp_type_codes"] = web_page_df["wp_type"].cat.codes.astype(
        codes_min_signed_type
    )
    web_page_newcols = ["wp_web_page_sk", "wp_type_codes"]
    web_page_df = web_page_df[web_page_newcols]

    web_clickstream_flist = glob.glob(config["data_dir"] + "web_clickstreams/*.parquet")

    task_ls = [
        delayed(etl_wcs)(
            fn, filtered_date_df.to_delayed()[0], web_page_df.to_delayed()[0]
        )
        for fn in web_clickstream_flist
    ]

    meta_d = {
        "wcs_user_sk": np.ones(1, dtype=np.int64),
        "tstamp_inSec": np.ones(1, dtype=np.int64),
        "wcs_sales_sk": np.ones(1, dtype=np.int64),
        "wp_type_codes": np.ones(1, dtype=np.int8),
    }
    meta_df = cudf.DataFrame(meta_d)
    merged_df = dask_cudf.from_delayed(task_ls, meta=meta_df)

    merged_df = merged_df.shuffle(on=["wcs_user_sk"])
    reviewed_sales = merged_df.map_partitions(
        reduction_function,
        REVIEW_CAT_CODE,
        meta=cudf.DataFrame({"wcs_sales_sk": np.ones(1, dtype=np.int64)}),
    )
    reviewed_sales = reviewed_sales.persist()
    wait(reviewed_sales)
    del merged_df

    all_sales_in_year = filtered_date_df.merge(
        web_sales_df, left_on=["d_date_sk"], right_on=["ws_sold_date_sk"], how="inner"
    )
    all_sales_in_year = all_sales_in_year[["ws_net_paid", "ws_order_number"]]

    all_sales_in_year = all_sales_in_year.persist()
    wait(all_sales_in_year)

    # note: switch to mainline
    # once https://github.com/dask/dask/pull/6066
    # lands

    q08_reviewed_sales = hash_merge(
        lhs=all_sales_in_year,
        rhs=reviewed_sales,
        left_on=["ws_order_number"],
        right_on=["wcs_sales_sk"],
        how="inner",
    )

    q08_reviewed_sales_sum = q08_reviewed_sales["ws_net_paid"].sum()
    q08_all_sales_sum = all_sales_in_year["ws_net_paid"].sum()

    q08_reviewed_sales_sum, q08_all_sales_sum = client.compute(
        [q08_reviewed_sales_sum, q08_all_sales_sum]
    )
    q08_reviewed_sales_sum, q08_all_sales_sum = (
        q08_reviewed_sales_sum.result(),
        q08_all_sales_sum.result(),
    )

    no_q08_review_sales_amount = q08_all_sales_sum - q08_reviewed_sales_sum

    final_result_df = cudf.DataFrame()
    final_result_df["q08_review_sales_amount"] = [q08_reviewed_sales_sum]
    final_result_df["q08_review_sales_amount"] = final_result_df[
        "q08_review_sales_amount"
    ].astype("int")
    final_result_df["no_q08_review_sales_amount"] = [no_q08_review_sales_amount]
    final_result_df["no_q08_review_sales_amount"] = final_result_df[
        "no_q08_review_sales_amount"
    ].astype("int")

    return final_result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
