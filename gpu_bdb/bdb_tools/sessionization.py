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

import cupy as cp
import numpy as np


def get_session_id_from_session_boundry(session_change_df, last_session_len):
    """
        This function returns session starts given a session change df
    """
    import cudf

    ## we dont really need the `session_id` to start from 0
    ## the total number of sessions per partition should be fairly limited
    ## and we really should not hit 2,147,483,647 sessions per partition
    ## Can switch to vec_arange code to match spark 1-1

    user_session_ids = cp.arange(len(session_change_df), dtype=np.int32)

    ### up shift the session length df
    session_len = session_change_df["t_index"].diff().reset_index(drop=True)
    session_len = session_len.shift(-1)
    session_len.iloc[-1] = last_session_len

    session_id_final_series = (
        cudf.Series(user_session_ids).repeat(session_len).reset_index(drop=True)
    )
    return session_id_final_series


def get_session_id(df, keep_cols, time_out):
    """
        This function creates a session id column for each click
        The session id grows in incremeant for each user's susbequent session
        Session boundry is defined by the time_out 
    """

    df["user_change_flag"] = df["wcs_user_sk"].diff(periods=1) != 0
    df["user_change_flag"] = df["user_change_flag"].fillna(True)
    df["session_timeout_flag"] = df["tstamp_inSec"].diff(periods=1) > time_out
    df["session_timeout_flag"] = df["session_timeout_flag"].fillna(False)

    df["session_change_flag"] = df["session_timeout_flag"] | df["user_change_flag"]

    # print(f"Total session change = {df['session_change_flag'].sum():,}")
    keep_cols = list(keep_cols)
    keep_cols += ["session_change_flag"]
    df = df[keep_cols]

    df = df.reset_index(drop=True)
    df["t_index"] = cp.arange(start=0, stop=len(df), dtype=np.int32)

    session_change_df = df[df["session_change_flag"]].reset_index(drop=True)
    last_session_len = len(df) - session_change_df["t_index"].iloc[-1]

    session_ids = get_session_id_from_session_boundry(
        session_change_df, last_session_len
    )

    assert len(session_ids) == len(df)
    return session_ids


def get_sessions(df, keep_cols, time_out=3600):
    df = df.sort_values(by=["wcs_user_sk", "tstamp_inSec"]).reset_index(drop=True)
    df["session_id"] = get_session_id(df, keep_cols, time_out)
    keep_cols += ["session_id"]
    df = df[keep_cols]
    return df


def get_distinct_sessions(df, keep_cols, time_out=3600):
    """
        ### Performence note
        The session + distinct 
        logic takes 0.2 seconds for a dataframe with 10M rows
        on gv-100
    """
    df = get_sessions(df, keep_cols, time_out=3600)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def get_pairs(
    df,
    merge_col=["session_id", "wcs_user_sk"],
    pair_col="i_category_id",
    output_col_1="category_id_1",
    output_col_2="category_id_2",
):
    """
        Gets pair after doing a inner merge
    """
    pair_df = df.merge(df, on=merge_col, suffixes=["_t1", "_t2"], how="inner")
    pair_df = pair_df[[f"{pair_col}_t1", f"{pair_col}_t2"]]
    pair_df = pair_df[
        pair_df[f"{pair_col}_t1"] < pair_df[f"{pair_col}_t2"]
    ].reset_index(drop=True)
    pair_df.columns = [output_col_1, output_col_2]
    return pair_df
