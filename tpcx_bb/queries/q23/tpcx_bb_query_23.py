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
import sys
import rmm


from xbb_tools.readers import build_reader
from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

from distributed import wait



### inventory date
q23_year = 2001
q23_month = 1
q23_coefficient = 1.3



def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"], basepath=config["data_dir"],
    )

    date_cols = ["d_date_sk", "d_year", "d_moy"]
    date_df = table_reader.read("date_dim", relevant_cols=date_cols)

    inv_cols = [
        "inv_warehouse_sk",
        "inv_item_sk",
        "inv_date_sk",
        "inv_quantity_on_hand",
    ]
    inv_df = table_reader.read("inventory", relevant_cols=inv_cols)

    return date_df, inv_df



def get_iteration1(merged_inv_dates, n_workers):
    grouped_df = merged_inv_dates.groupby(["inv_warehouse_sk", "inv_item_sk", "d_moy"])
    q23_tmp_inv_part = grouped_df.agg(
        {"inv_quantity_on_hand": ["mean", "std"]}, split_out=n_workers
    )
    q23_tmp_inv_part.columns = ["qty_mean", "qty_std"]
    q23_tmp_inv_part = q23_tmp_inv_part.reset_index(drop=False)
    q23_tmp_inv_part = q23_tmp_inv_part.loc[q23_tmp_inv_part["qty_mean"] > 0]
    q23_tmp_inv_part = q23_tmp_inv_part.reset_index(drop=True)

    q23_tmp_inv_part["qty_cov"] = (
        q23_tmp_inv_part["qty_std"] / q23_tmp_inv_part["qty_mean"]
    )

    iteration1_df = q23_tmp_inv_part.query(f"qty_cov >= {q23_coefficient}").reset_index(
        drop=True
    )

    return iteration1_df



def main(client,config):
    date_df, inv_df = benchmark(read_tables,config=config,compute_result=config["get_read_time"], dask_profile=config["dask_profile"])

    expr = (
        f"d_year == {q23_year} and (d_moy >= {q23_month} and d_moy <= {q23_month + 1})"
    )
    selected_dates_df = date_df.query(expr)

    merged_inv_dates = inv_df.merge(
        selected_dates_df, left_on="inv_date_sk", right_on="d_date_sk", how="inner"
    )
    n_workers = len(client.scheduler_info()["workers"])
    iteration1_df = get_iteration1(merged_inv_dates, n_workers)

    # Select only the columns we are interested in
    iteration1_df = iteration1_df[
        ["inv_warehouse_sk", "inv_item_sk", "d_moy", "qty_cov"]
    ].repartition(
        npartitions=1
    )  # iteration1_df has 40k rows at sf-100

    expr_1 = f"d_moy == {q23_month}"
    inv1_df = iteration1_df.query(expr_1)  # inv1_df has 13k rows at sf-100

    expr_2 = f"d_moy == {q23_month + 1}"
    inv2_df = iteration1_df.query(expr_2)  # 31k rows at sf-100

    result_df = inv1_df.merge(inv2_df, on=["inv_warehouse_sk", "inv_item_sk"])
    result_df = result_df.rename(
        columns={
            "d_moy_x": "d_moy",
            "d_moy_y": "inv2_d_moy",
            "qty_cov_x": "cov",
            "qty_cov_y": "inv2_cov",
        }
    )

    result_df = result_df.persist()

    result_df = result_df.sort_values(by=["inv_warehouse_sk", "inv_item_sk"])
    result_df = result_df.reset_index(drop=True)

    result_df = result_df.persist()
    wait(result_df)
    return result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
