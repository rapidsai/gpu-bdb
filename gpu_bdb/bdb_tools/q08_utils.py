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

import cupy as cp
import numpy as np

q08_SECONDS_BEFORE_PURCHASE = 259200
NA_FLAG = 0

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
    df["user_change_flag"] = df["user_change_flag"].fillna(True)
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

