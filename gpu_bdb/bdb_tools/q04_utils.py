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

import pandas as pd
import cudf

from bdb_tools.sessionization import get_sessions

from bdb_tools.readers import build_reader


def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
        backend=config["backend"],
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
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)

    if c:
        c.create_table('web_page_wo_categorical', wp_df, persist=False)
        c.create_table('web_clickstreams', wcs_df, persist=False)

    return wp_df, wcs_df


def abandonedShoppingCarts(df, DYNAMIC_CAT_CODE, ORDER_CAT_CODE):

    # Select groups where last dynamic row comes after last order row
    filtered_df = df[
        (df["wp_type_codes"] == ORDER_CAT_CODE)
        | (df["wp_type_codes"] == DYNAMIC_CAT_CODE)
    ]

    # Create a new column that is the concatenation of timestamp and wp_type_codes
    # (eg:123456:3, 234567:5)
    filtered_df["wp_type_codes"] = (
        filtered_df["tstamp_inSec"]
        .astype(str)
        .str.cat(filtered_df["wp_type_codes"].astype(str), sep=":")
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
    if isinstance(df, cudf.DataFrame):
        return cudf.DataFrame(
            {"pagecount": result.tstamp_inSec.sum(), "count": [len(result)]}
        )
    else:
        return pd.DataFrame(
            {"pagecount": result.tstamp_inSec.sum(), "count": [len(result)]}
        )

def reduction_function(df, keep_cols, DYNAMIC_CAT_CODE, ORDER_CAT_CODE):
    df = get_sessions(df, keep_cols=keep_cols)
    df = abandonedShoppingCarts(
        df, DYNAMIC_CAT_CODE=DYNAMIC_CAT_CODE, ORDER_CAT_CODE=ORDER_CAT_CODE
    )
    return df

