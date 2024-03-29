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

import cudf
import dask_cudf

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q03_utils import (
    apply_find_items_viewed,
    q03_purchased_item_IN,
    q03_purchased_item_category_IN,
    q03_limit,
    read_tables
)

from distributed import wait
import numpy as np

import glob
from dask import delayed

def get_wcs_minima(config):

    wcs_df = dask_cudf.read_parquet(
        os.path.join(config["data_dir"], "web_clickstreams/*.parquet"),
        columns=["wcs_click_date_sk", "wcs_click_time_sk"],
    )

    wcs_df["tstamp"] = wcs_df["wcs_click_date_sk"] * 86400 + wcs_df["wcs_click_time_sk"]

    wcs_tstamp_min = wcs_df["tstamp"].min().compute()
    return wcs_tstamp_min


def pre_repartition_task(wcs_fn, item_df, wcs_tstamp_min):

    wcs_cols = [
        "wcs_user_sk",
        "wcs_sales_sk",
        "wcs_item_sk",
        "wcs_click_date_sk",
        "wcs_click_time_sk",
    ]
    wcs_df = cudf.read_parquet(wcs_fn, columns=wcs_cols)
    wcs_df = wcs_df.dropna(axis=0, subset=["wcs_user_sk", "wcs_item_sk"])
    wcs_df["tstamp"] = wcs_df["wcs_click_date_sk"] * 86400 + wcs_df["wcs_click_time_sk"]
    wcs_df["tstamp"] = wcs_df["tstamp"] - wcs_tstamp_min

    wcs_df["tstamp"] = wcs_df["tstamp"].astype("int32")
    wcs_df["wcs_user_sk"] = wcs_df["wcs_user_sk"].astype("int32")
    wcs_df["wcs_sales_sk"] = wcs_df["wcs_sales_sk"].astype("int32")
    wcs_df["wcs_item_sk"] = wcs_df["wcs_item_sk"].astype("int32")

    merged_df = wcs_df.merge(
        item_df, left_on=["wcs_item_sk"], right_on=["i_item_sk"], how="inner"
    )

    del wcs_df
    del item_df

    cols_keep = [
        "wcs_user_sk",
        "tstamp",
        "wcs_item_sk",
        "wcs_sales_sk",
        "i_category_id",
    ]
    merged_df = merged_df[cols_keep]

    merged_df["wcs_user_sk"] = merged_df["wcs_user_sk"].astype("int32")
    merged_df["wcs_sales_sk"] = merged_df["wcs_sales_sk"].astype("int32")
    merged_df["wcs_item_sk"] = merged_df["wcs_item_sk"].astype("int32")
    merged_df["wcs_sales_sk"] = merged_df.wcs_sales_sk.fillna(0)
    return merged_df


def reduction_function(df, item_df_filtered):
    """
         Combines all the reduction ops into a single frame
    """
    product_view_results = apply_find_items_viewed(df, item_mappings=item_df_filtered)

    grouped_df = product_view_results.groupby(["i_item_sk"]).size().reset_index()
    grouped_df.columns = ["i_item_sk", "cnt"]
    return grouped_df


def main(client, config):

    item_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

    wcs_tstamp_min = get_wcs_minima(config)

    item_df["i_item_sk"] = item_df["i_item_sk"].astype("int32")
    item_df["i_category_id"] = item_df["i_category_id"].astype("int8")

    # we eventually will only care about these categories, so we can filter now
    item_df_filtered = item_df.loc[
        item_df.i_category_id.isin(q03_purchased_item_category_IN)
    ].reset_index(drop=True)

    # The main idea is that we don't fuse a filtration task with reading task yet
    # this causes more memory pressures as we try to read the whole thing ( and spill that)
    # at once and then do filtration .

    web_clickstream_flist = glob.glob(os.path.join(config["data_dir"], "web_clickstreams/*.parquet"))
    task_ls = [
        delayed(pre_repartition_task)(fn, item_df.to_delayed()[0], wcs_tstamp_min)
        for fn in web_clickstream_flist
    ]

    meta_d = {
        "wcs_user_sk": np.ones(1, dtype=np.int32),
        "tstamp": np.ones(1, dtype=np.int32),
        "wcs_item_sk": np.ones(1, dtype=np.int32),
        "wcs_sales_sk": np.ones(1, dtype=np.int32),
        "i_category_id": np.ones(1, dtype=np.int8),
    }
    meta_df = cudf.DataFrame(meta_d)

    merged_df = dask_cudf.from_delayed(task_ls, meta=meta_df)

    merged_df = merged_df.shuffle(on="wcs_user_sk")

    meta_d = {
        "i_item_sk": np.ones(1, dtype=merged_df["wcs_item_sk"].dtype),
        "cnt": np.ones(1, dtype=merged_df["wcs_item_sk"].dtype),
    }
    meta_df = cudf.DataFrame(meta_d)

    grouped_df = merged_df.map_partitions(
        reduction_function, item_df_filtered.to_delayed()[0], meta=meta_df
    )

    ### todo: check if this has any impact on stability
    grouped_df = grouped_df.persist(priority=10000)
    ### todo: remove this later after more testing
    wait(grouped_df)
    print("---" * 20)
    print("grouping complete ={}".format(len(grouped_df)))
    grouped_df = grouped_df.groupby(["i_item_sk"]).sum(split_every=2).reset_index()
    grouped_df.columns = ["i_item_sk", "cnt"]
    result_df = grouped_df.map_partitions(
        lambda df: df.sort_values(by=["cnt"], ascending=False)
    )

    result_df.columns = ["lastviewed_item", "cnt"]
    result_df["purchased_item"] = q03_purchased_item_IN
    cols_order = ["purchased_item", "lastviewed_item", "cnt"]
    result_df = result_df[cols_order]
    result_df = result_df.persist()
    ### todo: remove this later after more testing
    wait(result_df)
    print(len(result_df))
    result_df = result_df.head(q03_limit)
    print("result complete")
    print("---" * 20)
    return result_df


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
