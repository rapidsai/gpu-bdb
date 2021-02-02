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

from dask.distributed import Client

import numpy as np
import sys


from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)
from bdb_tools.readers import build_reader


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    ws_columns = ["ws_ship_hdemo_sk", "ws_web_page_sk", "ws_sold_time_sk"]
    web_sales = table_reader.read("web_sales", relevant_cols=ws_columns)

    hd_columns = ["hd_demo_sk", "hd_dep_count"]
    household_demographics = table_reader.read(
        "household_demographics", relevant_cols=hd_columns
    )

    wp_columns = ["wp_web_page_sk", "wp_char_count"]
    web_page = table_reader.read("web_page", relevant_cols=wp_columns)

    td_columns = ["t_time_sk", "t_hour"]
    time_dim = table_reader.read("time_dim", relevant_cols=td_columns)

    return web_sales, household_demographics, web_page, time_dim


def main(client, config):
    import cudf

    q14_dependents = 5
    q14_morning_startHour = 7
    q14_morning_endHour = 8
    q14_evening_startHour = 19
    q14_evening_endHour = 20
    q14_content_len_min = 5000
    q14_content_len_max = 6000

    web_sales, household_demographics, web_page, time_dim = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    household_demographics = household_demographics.query(
        "hd_dep_count==@q14_dependents",
        meta=household_demographics._meta,
        local_dict={"q14_dependents": q14_dependents},
    ).reset_index(drop=True)
    output_table = web_sales.merge(
        household_demographics,
        left_on=["ws_ship_hdemo_sk"],
        right_on=["hd_demo_sk"],
        how="inner",
    )

    output_table = output_table.drop(
        columns=["ws_ship_hdemo_sk", "hd_demo_sk", "hd_dep_count"]
    )

    web_page = web_page.query(
        "wp_char_count>=@q14_content_len_min and wp_char_count<=@q14_content_len_max",
        meta=web_page._meta,
        local_dict={
            "q14_content_len_min": q14_content_len_min,
            "q14_content_len_max": q14_content_len_max,
        },
    ).reset_index(drop=True)
    output_table = output_table.merge(
        web_page, left_on=["ws_web_page_sk"], right_on=["wp_web_page_sk"], how="inner"
    )

    output_table = output_table.drop(
        columns=["ws_web_page_sk", "wp_web_page_sk", "wp_char_count"]
    )

    time_dim = time_dim.query(
        "t_hour==@q14_morning_startHour or t_hour==@q14_morning_endHour or t_hour==@q14_evening_startHour or t_hour==@q14_evening_endHour",
        meta=time_dim._meta,
        local_dict={
            "q14_morning_startHour": q14_morning_startHour,
            "q14_morning_endHour": q14_morning_endHour,
            "q14_evening_startHour": q14_evening_startHour,
            "q14_evening_endHour": q14_evening_endHour,
        },
    )
    output_table = output_table.merge(
        time_dim, left_on=["ws_sold_time_sk"], right_on=["t_time_sk"], how="inner"
    )
    output_table = output_table.drop(columns=["ws_sold_time_sk", "t_time_sk"])

    output_table["am"] = (output_table["t_hour"] >= q14_morning_startHour) & (
        output_table["t_hour"] <= q14_morning_endHour
    ).reset_index(drop=True)
    output_table["pm"] = (output_table["t_hour"] >= q14_evening_startHour) & (
        output_table["t_hour"] <= q14_evening_endHour
    ).reset_index(drop=True)

    am_pm_ratio = output_table["am"].sum() / output_table["pm"].sum()
    # result is a scalor
    am_pm_ratio = am_pm_ratio.persist()
    result = am_pm_ratio.compute()
    if np.isinf(result):
        result = -1.0

    print(result)
    # result is a scalor
    result_df = cudf.DataFrame({"am_pm_ratio": result})

    return result_df


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
