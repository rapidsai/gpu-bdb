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

import os
import glob

import cudf
import dask_cudf

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
    convert_datestring_to_days
)
from bdb_tools.merge_util import hash_merge
from bdb_tools.q08_utils import (
    get_sessions,
    get_unique_sales_keys_from_sessions,
    prep_for_sessionization,
    q08_STARTDATE,
    q08_ENDDATE,
    read_tables
)

import numpy as np
from distributed import wait
from dask import delayed

def etl_wcs(wcs_fn, filtered_date_df, web_page_df):

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


def reduction_function(df, REVIEW_CAT_CODE):

    # category code of review records
    df["review_flag"] = df.wp_type_codes == REVIEW_CAT_CODE

    # set_index in the previous statement will make sure all records of each wcs_user_sk end up in one partition.
    df = prep_for_sessionization(df, review_cat_code=REVIEW_CAT_CODE)
    df = get_sessions(df)
    df = get_unique_sales_keys_from_sessions(df, REVIEW_CAT_CODE)
    return df.to_frame()


def main(client, config):

    (date_dim_df, web_page_df, web_sales_df) = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

    date_meta_df = date_dim_df._meta
    date_meta_df["d_date"] = date_meta_df["d_date"].astype("int64")
    date_dim_cov_df = date_dim_df.map_partitions(convert_datestring_to_days, meta=date_meta_df)
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

    web_clickstream_flist = glob.glob(os.path.join(config["data_dir"], "web_clickstreams/*.parquet"))

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
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
