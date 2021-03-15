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
from collections import OrderedDict


from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
    convert_datestring_to_days,
)
from bdb_tools.readers import build_reader

import datetime
import numpy as np


q15_startDate = "2001-09-02"
q15_endDate = "2002-09-02"
q15_store_sk = "10"

store_sales_cols = ["ss_sold_date_sk", "ss_net_paid", "ss_store_sk", "ss_item_sk"]
date_cols = ["d_date", "d_date_sk"]
item_cols = ["i_item_sk", "i_category_id"]


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
    item_df = table_reader.read("item", relevant_cols=item_cols)

    return store_sales_df, date_dim_df, item_df


def main(client, config):

    store_sales_df, date_dim_df, item_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    ###  Query 0. Filtering store sales
    store_sales_df = store_sales_df.query(f"ss_store_sk == {q15_store_sk}")

    ### Query 1. Date Time Filteration Logic
    date_dim_cov_df = date_dim_df.map_partitions(convert_datestring_to_days)

    q15_start_dt = datetime.datetime.strptime(q15_startDate, "%Y-%m-%d")
    q15_start_dt = (q15_start_dt - datetime.datetime(1970, 1, 1)) / datetime.timedelta(
        days=1
    )
    q15_start_dt = int(q15_start_dt)

    q15_end_dt = datetime.datetime.strptime(q15_endDate, "%Y-%m-%d")
    q15_end_dt = (q15_end_dt - datetime.datetime(1970, 1, 1)) / datetime.timedelta(
        days=1
    )
    q15_end_dt = int(q15_end_dt)

    filtered_date_df = date_dim_cov_df.query(
        f"d_date >={q15_start_dt} and d_date <= {q15_end_dt}",
        meta=date_dim_cov_df._meta,
    ).reset_index(drop=True)

    store_sales_df = store_sales_df.merge(
        filtered_date_df,
        left_on=["ss_sold_date_sk"],
        right_on=["d_date_sk"],
        how="inner",
    )
    store_sales_df = store_sales_df[store_sales_cols]

    #### Query 2. `store_sales_df` inner join `item`
    item_df = item_df[item_df["i_category_id"].notnull()].reset_index(drop=True)
    store_sales_item_join = store_sales_df.merge(
        item_df, left_on=["ss_item_sk"], right_on=["i_item_sk"], how="inner"
    )

    group_cols = ["i_category_id", "ss_sold_date_sk"]
    agg_cols = ["ss_net_paid"]

    agg_df = (
        store_sales_item_join[agg_cols + group_cols]
        .groupby(group_cols)
        .agg({"ss_net_paid": "sum"})
    )
    ### The number of categories is know to be limited
    agg_df = agg_df.compute()

    agg_df = agg_df.reset_index(drop=False)
    agg_df = agg_df.rename(
        columns={"i_category_id": "cat", "ss_sold_date_sk": "x", "ss_net_paid": "y"}
    )
    agg_df["xy"] = agg_df["x"] * agg_df["y"]
    agg_df["xx"] = agg_df["x"] * agg_df["x"]

    ### Query 3. Group By Logic

    # Ideally we should use `as_index=False` and have a simplified rename call.
    # as_index=False doesn't work here: https://github.com/rapidsai/cudf/issues/3737
    regression_groups = agg_df.groupby(["cat"]).agg(
        {"x": ["count", "sum"], "xx": ["sum"], "xy": ["sum"], "y": ["count", "sum"]}
    )
    regression_groups.columns = regression_groups.columns.map(
        {
            ("x", "count"): "count_x",
            ("x", "sum"): "sum_x",
            ("xx", "sum"): "sum_xx",
            ("xy", "sum"): "sum_xy",
            ("y", "count"): "count_y",
            ("y", "sum"): "sum_y",
        }
    )

    regression_groups["slope"] = (
        regression_groups["count_x"] * regression_groups["sum_xy"]
        - regression_groups["sum_x"] * regression_groups["sum_y"]
    ) / (
        (
            regression_groups["count_x"] * regression_groups["sum_xx"]
            - regression_groups["sum_x"] * regression_groups["sum_x"]
        )
    )

    ### Applying Regression Formula

    regression_groups["intercept"] = (
        regression_groups["sum_y"]
        - regression_groups["slope"] * regression_groups["sum_x"]
    )
    regression_groups["intercept"] = (
        regression_groups["intercept"] / regression_groups["count_x"]
    )

    regression_groups = regression_groups[regression_groups["slope"] < 0]
    regression_groups = regression_groups.reset_index(drop=False)
    regression_groups = regression_groups[["cat", "slope", "intercept"]].sort_values(
        by=["cat"]
    )

    return regression_groups


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
