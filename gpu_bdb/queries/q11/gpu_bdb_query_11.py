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

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
    convert_datestring_to_days,
)

from bdb_tools.q11_utils import read_tables

import numpy as np

q11_start_date = "2003-01-02"
q11_end_date = "2003-02-02"

def main(client, config):

    pr_df, ws_df, date_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    date_df = date_df.map_partitions(convert_datestring_to_days)

    # Filter limit in days
    min_date = np.datetime64(q11_start_date, "D").astype(int)
    max_date = np.datetime64(q11_end_date, "D").astype(int)

    date_df = date_df.query(
        "d_date>=@min_date and d_date<=@max_date",
        meta=date_df._meta,
        local_dict={"min_date": min_date, "max_date": max_date},
    )

    web_sales = ws_df.merge(
        date_df, left_on=["ws_sold_date_sk"], right_on=["d_date_sk"], how="inner"
    )

    # Only take sales that are not null, and get the unique item SKUs
    # Note that we do not need the revenue column, so we don't need a groupby aggregation
    # Spark *possibly* does this optimization under the hood
    web_sales = web_sales[web_sales["ws_item_sk"].notnull()].reset_index(drop=True)
    web_sales = web_sales.ws_item_sk.unique().to_frame()

    # temporarily avoid reset_index due to a MultiColumn bug
    product_reviews = pr_df[pr_df["pr_item_sk"].notnull()].reset_index(drop=True)
    product_reviews = product_reviews.groupby("pr_item_sk").agg(
        {"pr_review_rating": ["count", "mean"]}
    )
    product_reviews.columns = ["r_count", "avg_rating"]

    # temporarily avoid reset_index due to a MultiColumn bug
    sales = web_sales.merge(
        product_reviews, left_on=["ws_item_sk"], right_index=True, how="inner"
    )

    sales = sales.rename(
        columns={
            "ws_item_sk": "pid",
            "r_count": "reviews_count",
            "revenue": "m_revenue",
        }
    ).reset_index()

    # this is a scalar so can remain a cudf frame
    sales_corr = sales["reviews_count"].corr(sales["avg_rating"])
    sales_corr = sales_corr.persist()
    sales_corr = sales_corr.compute()
    result_df = cudf.DataFrame([sales_corr])
    result_df.columns = ["corr(CAST(reviews_count AS DOUBLE), avg_rating)"]

    return result_df


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
