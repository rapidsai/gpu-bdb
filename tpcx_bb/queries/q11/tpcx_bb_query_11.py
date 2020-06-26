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
    run_dask_cudf_query,
    convert_datestring_to_days,
)
from xbb_tools.readers import build_reader

from numba import cuda
import numpy as np

cli_args = tpcxbb_argparser()

q11_start_date = "2003-01-02"
q11_end_date = "2003-02-02"


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables():
    table_reader = build_reader(
        cli_args["file_format"],
        basepath=cli_args["data_dir"],
        repartition_small_table=cli_args["repartition_small_table"],
        split_row_groups=cli_args["split_row_groups"],
    )

    product_review_cols = [
        "pr_review_rating",
        "pr_item_sk",
    ]
    web_sales_cols = [
        "ws_sold_date_sk",
        "ws_net_paid",
        "ws_item_sk",
    ]
    date_cols = ["d_date_sk", "d_date"]

    pr_df = table_reader.read("product_reviews", relevant_cols=product_review_cols)
    # we only read int columns here so it should scale up to sf-10k as just 26M rows
    pr_df = pr_df.repartition(npartitions=1)

    ws_df = table_reader.read("web_sales", relevant_cols=web_sales_cols)
    date_df = table_reader.read("date_dim", relevant_cols=date_cols)

    return pr_df, ws_df, date_df


@benchmark(dask_profile=cli_args["dask_profile"])
def main(client):
    import cudf

    pr_df, ws_df, date_df = read_tables()

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
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client = attach_to_cluster(cli_args)

    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
