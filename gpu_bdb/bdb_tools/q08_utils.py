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
import pandas as pd

import cupy as cp
import numpy as np

from bdb_tools.readers import build_reader

q08_STARTDATE = "2001-09-02"
q08_ENDDATE = "2002-09-02"
q08_SECONDS_BEFORE_PURCHASE = 259200
NA_FLAG = 0

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
        backend=config["backend"],
    )

    date_dim_cols = ["d_date_sk", "d_date"]
    web_page_cols = ["wp_web_page_sk", "wp_type"]
    web_sales_cols = ["ws_net_paid", "ws_order_number", "ws_sold_date_sk"]
    wcs_cols = [
        "wcs_user_sk",
        "wcs_sales_sk",
        "wcs_click_date_sk",
        "wcs_click_time_sk",
        "wcs_web_page_sk",
    ]

    date_dim_df = table_reader.read("date_dim", relevant_cols=date_dim_cols)
    web_page_df = table_reader.read("web_page", relevant_cols=web_page_cols)
    web_sales_df = table_reader.read("web_sales", relevant_cols=web_sales_cols)
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)

    if c:
        c.create_table("web_clickstreams", wcs_df, persist=False)
        c.create_table("web_sales", web_sales_df, persist=False)
        c.create_table("web_page", web_page_df, persist=False)
        c.create_table("date_dim", date_dim_df, persist=False)

    return (date_dim_df, web_page_df, web_sales_df)

def get_session_id_from_session_boundary(session_change_df, last_session_len):
    """
        This function returns session starts given a session change df
    """

    user_session_ids = session_change_df.tstamp_inSec

    ### up shift the session length df
    session_len = session_change_df["t_index"].diff().reset_index(drop=True)
    session_len = session_len.shift(-1)

    try:
        session_len.iloc[-1] = last_session_len
    except (AssertionError, IndexError):  # IndexError in numba >= 0.48
        if isinstance(session_change_df, cudf.DataFrame):
            session_len = cudf.Series([])
        else:
            session_len = pd.Series([]) 
   
    if isinstance(session_change_df, cudf.DataFrame):
        session_id_final_series = (
            cudf.Series(user_session_ids).repeat(session_len).reset_index(drop=True)
        )
    else:
        session_id_final_series = (
           pd.Series(user_session_ids).repeat(session_len).reset_index(drop=True)
        )
    return session_id_final_series


def get_session_id(df):
    """
        This function creates a session id column for each click
        The session id grows in incremeant for each user's susbequent session
        Session boundry is defined by the time_out
    """

    df["user_change_flag"] = df["wcs_user_sk"].diff(periods=1) != 0
    df["user_change_flag"] = df["user_change_flag"].fillna(True)
    df["session_change_flag"] = df["review_flag"] | df["user_change_flag"]

    df = df.reset_index(drop=True)
    
    if isinstance(df, cudf.DataFrame):
        df["t_index"] = cp.arange(start=0, stop=len(df), dtype=np.int32)
    else:
        df["t_index"] = np.arange(start=0, stop=len(df), dtype=np.int32)
        
    session_change_df = df[df["session_change_flag"]].reset_index(drop=True)
    try:
        last_session_len = len(df) - session_change_df["t_index"].iloc[-1]
    except (AssertionError, IndexError):  # IndexError in numba >= 0.48
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

