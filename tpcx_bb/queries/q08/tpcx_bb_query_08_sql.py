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
import cupy as cp
import numpy as np
import cudf

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

from dask.distributed import wait

# -------- Q8 -----------
q08_SECONDS_BEFORE_PURCHASE = 259200
q08_STARTDATE = "2001-09-02"
q08_ENDDATE = "2002-09-02"

REVIEW_CAT_CODE = 6
NA_FLAG = 0


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


def read_tables(data_dir, bc):
    bc.create_table("web_clickstreams", data_dir + "/web_clickstreams/*.parquet")
    bc.create_table("web_sales", data_dir + "/web_sales/*.parquet")
    bc.create_table("web_page", data_dir + "/web_page/*.parquet")
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query_1 = f"""
        SELECT d_date_sk
        FROM date_dim
        WHERE CAST(d_date as date) IN (date '{q08_STARTDATE}',
                                       date '{q08_ENDDATE}')
        ORDER BY CAST(d_date as date) asc
    """
    result_dates_sk_filter = bc.sql(query_1).compute()

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
    web_page_df = bc.sql(query_aux)

    # cast to minimum viable dtype
    web_page_df["wp_type"] = web_page_df["wp_type"].map_partitions(
        lambda ser: ser.astype("category")
    )

    cpu_categories = web_page_df["wp_type"].compute().cat.categories.to_pandas()
    REVIEW_CAT_CODE = cpu_categories.get_loc("review")

    codes_min_signed_type = cudf.utils.dtypes.min_signed_type(len(cpu_categories))

    web_page_df["wp_type_codes"] = web_page_df["wp_type"].cat.codes.astype(
        codes_min_signed_type
    )

    web_page_newcols = ["wp_web_page_sk", "wp_type_codes"]
    web_page_df = web_page_df[web_page_newcols]

    web_page_df = web_page_df.persist()
    wait(web_page_df)
    bc.create_table('web_page_2', web_page_df)

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
        ORDER BY wcs_user_sk
    """
    merged_df = bc.sql(query_2)

    bc.drop_table("web_page_2")
    del web_page_df

    merged_df = merged_df.repartition(columns=["wcs_user_sk"])
    merged_df["review_flag"] = merged_df.wp_type_codes == REVIEW_CAT_CODE

    prepped = merged_df.map_partitions(
        prep_for_sessionization, review_cat_code=REVIEW_CAT_CODE
    )

    sessionized = prepped.map_partitions(get_sessions)

    unique_review_sales = sessionized.map_partitions(
        get_unique_sales_keys_from_sessions, review_cat_code=REVIEW_CAT_CODE
    )
    
    unique_review_sales = unique_review_sales.to_frame()

    unique_review_sales = unique_review_sales.persist()
    wait(unique_review_sales)
    bc.create_table("reviews", unique_review_sales)
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
    result = bc.sql(last_query)

    bc.drop_table("reviews")
    return result


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
