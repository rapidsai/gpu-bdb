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

import inspect
import os
import shutil
import socket
import re
import argparse
import time
import subprocess
from datetime import datetime
from collections.abc import Iterable
import glob
import dask
import traceback
import yaml
import sys
from collections import OrderedDict
from collections.abc import MutableMapping

import numpy as np

import pandas as pd
import dask.dataframe as dd
from dask.utils import parse_bytes
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait, performance_report, SSHCluster

import json

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from xbb_tools.cluster_startup import get_config_options

#################################
# Benchmark Timing
#################################
def benchmark(func, *args, **kwargs):
    csv = kwargs.pop("csv", True)
    dask_profile = kwargs.pop("dask_profile", False)
    compute_result = kwargs.pop("compute_result", False)
    name = func.__name__
    t0 = time.time()
    if dask_profile:
        with performance_report(filename=f"profiled-{name}.html"):
            result = func(*args, **kwargs)
    else:
        result = func(*args, **kwargs)
    elapsed_time = time.time() - t0

    logging_info = {}
    logging_info["elapsed_time_seconds"] = elapsed_time
    logging_info["function_name"] = name
    if compute_result:
        import dask_cudf

        if isinstance(result, dask_cudf.DataFrame):
            len_tasks = [dask.delayed(len)(df) for df in result.to_delayed()]
        else:
            len_tasks = []
            for read_df in result:
                len_tasks += [dask.delayed(len)(df) for df in read_df.to_delayed()]

        compute_st = time.time()
        results = dask.compute(*len_tasks)
        compute_et = time.time()
        logging_info["compute_time_seconds"] = compute_et - compute_st

    logdf = pd.DataFrame.from_dict(logging_info, orient="index").T

    if csv:
        logdf.to_csv(f"benchmarked_{name}.csv", index=False)
    else:
        print(logdf)
    return result


#################################
# Result Writing
#################################


def write_result(payload, filetype="parquet", output_directory="./"):
    """
    """
    import cudf

    if isinstance(payload, MutableMapping):
        if payload.get("output_type", None) == "supervised":
            write_supervised_learning_result(
                result_dict=payload,
                filetype=filetype,
                output_directory=output_directory,
            )
        else:
            write_clustering_result(
                result_dict=payload,
                filetype=filetype,
                output_directory=output_directory,
            )
    elif isinstance(payload, cudf.DataFrame) or isinstance(payload, dd.DataFrame):
        write_etl_result(
            df=payload, filetype=filetype, output_directory=output_directory
        )
    else:
        raise ValueError("payload must be a dict or a dataframe.")


def write_etl_result(df, filetype="parquet", output_directory="./"):
    assert filetype in ["csv", "parquet"]

    QUERY_NUM = get_query_number()
    if filetype == "csv":
        output_path = f"{output_directory}q{QUERY_NUM}-results.csv"

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        df.to_csv(output_path, header=True, index=False)
    else:
        output_path = f"{output_directory}q{QUERY_NUM}-results.parquet"
        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                ## to remove existing  directory
                shutil.rmtree(output_path)
            else:
                ## to remove existing single parquet file
                os.remove(output_path)

        if isinstance(df, dd.DataFrame):
            df.to_parquet(output_path, write_index=False)

        else:
            df.to_parquet(
                f"{output_directory}q{QUERY_NUM}-results.parquet", index=False
            )


def write_result_q05(results_dict, output_directory="./", filetype=None):
    """
    Results are a text file due to the structure and tiny size
    Filetype argument added for compatibility. Is not used.
    """
    with open(f"{output_directory}q05-metrics-results.txt", "w") as outfile:
        outfile.write("Precision: %f\n" % results_dict["precision"])
        outfile.write("AUC: %f\n" % results_dict["auc"])
        outfile.write("Confusion Matrix:\n")
        cm = results_dict["confusion_matrix"]
        outfile.write(
            "%8.1f  %8.1f\n%8.1f %8.1f\n" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
        )


def write_supervised_learning_result(result_dict, output_directory, filetype="csv"):
    assert filetype in ["csv", "parquet"]
    QUERY_NUM = get_query_number()
    if QUERY_NUM == "05":
        write_result_q05(result_dict, output_directory)
    else:
        df = result_dict["df"]
        acc = result_dict["acc"]
        prec = result_dict["prec"]
        cmat = result_dict["cmat"]

        with open(f"{output_directory}q{QUERY_NUM}-metrics-results.txt", "w") as out:
            out.write("Precision: %s\n" % prec)
            out.write("Accuracy: %s\n" % acc)
            out.write(
                "Confusion Matrix: \n%s\n"
                % (str(cmat).replace("[", " ").replace("]", " ").replace(".", ""))
            )

        if filetype == "csv":
            df.to_csv(
                f"{output_directory}q{QUERY_NUM}-results.csv", header=False, index=None
            )
        else:
            df.to_parquet(
                f"{output_directory}q{QUERY_NUM}-results.parquet", write_index=False
            )


def write_clustering_result(result_dict, output_directory="./", filetype="csv"):
    """Results are a text file AND a csv or parquet file.
    This works because we are down to a single partition dataframe.
    """
    assert filetype in ["csv", "parquet"]

    QUERY_NUM = get_query_number()
    clustering_info_name = f"{QUERY_NUM}-results-cluster-info.txt"

    with open(f"{output_directory}q{clustering_info_name}", "w") as fh:
        fh.write("Clusters:\n\n")
        fh.write(f"Number of Clusters: {result_dict.get('nclusters')}\n")
        fh.write(f"WSSSE: {result_dict.get('wssse')}\n")

        centers = result_dict.get("cluster_centers")
        for center in centers.values.tolist():
            fh.write(f"{center}\n")

    # this is a single partition dataframe, with cid_labels hard coded
    # as the label column
    data = result_dict.get("cid_labels")

    if filetype == "csv":
        clustering_result_name = f"q{QUERY_NUM}-results.csv"
        data.to_csv(
            f"{output_directory}{clustering_result_name}", index=False, header=None
        )
    else:
        clustering_result_name = f"q{QUERY_NUM}-results.parquet"
        data.to_parquet(f"{output_directory}{clustering_result_name}", index=False)

    return 0


def remove_benchmark_files():
    """
    Removes benchmark result files from cwd
    to ensure that we don't upload stale results
    """
    fname_ls = [
        "benchmarked_write_result.csv",
        "benchmarked_read_tables.csv",
        "benchmarked_main.csv",
    ]
    for fname in fname_ls:
        if os.path.exists(fname):
            os.remove(fname)


#################################
# Query Runner Utilities
#################################
def run_query(
    config, client, query_func, write_func=write_result, blazing_context=None
):
    if blazing_context:
        run_bsql_query(
            config=config,
            client=client,
            query_func=query_func,
            blazing_context=blazing_context,
            write_func=write_func,
        )
    else:
        run_dask_cudf_query(
            config=config, client=client, query_func=query_func, write_func=write_func,
        )


def run_dask_cudf_query(config, client, query_func, write_func=write_result):
    """
    Common utility to perform all steps needed to execute a dask-cudf version
    of the query. Includes attaching to cluster, running the query and writing results
    """
    try:
        remove_benchmark_files()
        config["start_time"] = time.time()
        results = benchmark(
            query_func,
            dask_profile=config.get("dask_profile"),
            client=client,
            config=config,
        )

        benchmark(
            write_func,
            results,
            output_directory=config["output_dir"],
            filetype=config["output_filetype"],
        )
        config["query_status"] = "Success"

        result_verified = False

        if config["verify_results"]:
            result_verified = verify_results(config["verify_dir"])
        config["result_verified"] = result_verified

    except:
        config["query_status"] = "Failed"
        print("Encountered Exception while running query")
        print(traceback.format_exc())

    # google sheet benchmarking automation
    push_payload_to_googlesheet(config)


def run_bsql_query(
    config, client, query_func, blazing_context, write_func=write_result
):
    """
    Common utility to perform all steps needed to execute a dask-cudf version
    of the query. Includes attaching to cluster, running the query and writing results
    """
    # TODO: Unify this with dask-cudf version
    try:
        remove_benchmark_files()
        config["start_time"] = time.time()
        data_dir = config["data_dir"]
        results = benchmark(
            query_func,
            dask_profile=config.get("dask_profile"),
            data_dir=data_dir,
            client=client,
            bc=blazing_context,
            config=config,
        )

        benchmark(
            write_func,
            results,
            output_directory=config["output_dir"],
            filetype=config["output_filetype"],
        )
        config["query_status"] = "Success"

        result_verified = False

        if config["verify_results"]:
            result_verified = verify_results(config["verify_dir"])
        config["result_verified"] = result_verified

    except:
        config["query_status"] = "Failed"
        print("Encountered Exception while running query")
        print(traceback.format_exc())

    # google sheet benchmarking automation
    push_payload_to_googlesheet(config)


def add_empty_config(args):
    keys = [
        "get_read_time",
        "split_row_groups",
        "dask_profile",
        "verify_results",
    ]

    for key in keys:
        if key not in args:
            args[key] = None

    if "file_format" not in args:
        args["file_format"] = "parquet"

    if "output_filetype" not in args:
        args["output_filetype"] = "parquet"

    return args


def tpcxbb_argparser():
    args = get_tpcxbb_argparser_commandline_args()
    with open(args["config_file"]) as fp:
        args = yaml.safe_load(fp.read())
    args = add_empty_config(args)

    return args


def get_tpcxbb_argparser_commandline_args():
    parser = argparse.ArgumentParser(description="Run TPCx-BB query")
    print("Using default arguments")
    parser.add_argument(
        "--config_file",
        default="benchmark_runner/benchmark_config.yaml",
        type=str,
        help="Location of benchmark configuration yaml file",
    )

    args = parser.parse_args()
    args = vars(args)
    return args


def get_scale_factor(data_dir):
    """
        Returns scale factor from data_dir
    """
    reg_match = re.search("sf[0-9]+", data_dir).group(0)
    return int(reg_match[2:])


def get_query_number():
    """This assumes a directory structure like:
    - rapids-queries
        - q01
        - q02
        ...
    """
    QUERY_NUM = os.getcwd().split("/")[-1].strip("q")
    return QUERY_NUM


#################################
# Correctness Verification
#################################


def assert_dataframes_pseudo_equal(df1, df2, significant=6):
    """Verify the pseudo-equality of two dataframes, acknowledging that:
        - Row ordering may not be consistent between files
        - Column ordering may vary between files,
        - Floating point math can be annoying, so we may need to assert
            equality at a specified level of precision

    and assuming that:
        - Files do not contain their own index values
        - Column presence does not vary between files
        - Datetime columns are read into memory consistently as either Object or Datetime columns
    """
    from cudf.tests.utils import assert_eq

    # check shape is the same
    assert df1.shape == df2.shape

    # check columns are the same
    assert sorted(df1.columns.tolist()) == sorted(df2.columns.tolist())

    # align column ordering across dataframes
    df2 = df2[df1.columns]

    # sort by every column, with the stable column ordering, then reset the index
    df1 = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2 = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)

    # verify equality
    assert_eq(df1, df2, check_less_precise=significant, check_dtype=False)


def calculate_label_overlap_percent(spark_labels, rapids_labels):

    assert len(spark_labels) == len(rapids_labels)

    spark_labels.columns = ["cid", "label"]
    rapids_labels.columns = ["cid", "label"]

    # assert that we clustered the same IDs
    assert spark_labels.cid.equals(rapids_labels.cid)

    rapids_counts_normalized = rapids_labels.label.value_counts(
        normalize=True
    ).reset_index()
    spark_counts_normalized = spark_labels.label.value_counts(
        normalize=True
    ).reset_index()

    nclusters = 8
    label_mapping = {}

    for i in range(nclusters):
        row_spark = spark_counts_normalized.iloc[i]
        row_rapids = rapids_counts_normalized.iloc[i]

        percent = row_spark["label"]
        label_id_spark = row_spark["index"]
        label_id_rapids = row_rapids["index"]

        label_mapping[label_id_rapids.astype("int")] = label_id_spark.astype("int")

    rapids_labels["label"] = rapids_labels["label"].replace(label_mapping)
    merged = spark_labels.merge(rapids_labels, how="inner", on=["cid"])
    overlap_percent = (merged.label_x == merged.label_y).sum() / len(merged) * 100
    return overlap_percent


def compare_clustering_cost(spark_path, rapids_path):
    with open(spark_path, "r") as fh:
        spark_results = fh.readlines()

    with open(rapids_path, "r") as fh:
        rapids_results = fh.readlines()

    spark_wssse = float(spark_results[3].split(": ")[1])
    rapids_wssse = float(rapids_results[3].split(": ")[1])

    delta_percent = abs(spark_wssse - rapids_wssse) / spark_wssse * 100

    tolerance = 0.01  # allow for 1/100th of a percent cost difference
    rapids_cost_similar = (rapids_wssse <= spark_wssse) or (delta_percent <= tolerance)

    print(f"Cost delta percent: {delta_percent}")
    print(f"RAPIDS cost lower/similar: {rapids_cost_similar}")
    return rapids_cost_similar, delta_percent


def verify_clustering_query_cost(spark_path, rapids_path):
    rapids_cost_lower, delta_percent = compare_clustering_cost(spark_path, rapids_path,)
    assert rapids_cost_lower


def verify_clustering_query_labels(spark_data, rapids_data):
    overlap_percent = calculate_label_overlap_percent(spark_data, rapids_data)
    print(f"Label overlap percent: {overlap_percent}")
    return 0


def compare_supervised_metrics(validation, results):
    val_precision = float(validation[0].split(": ")[1])
    val_auc = float(validation[1].split(": ")[1])

    results_precision = float(results[0].split(": ")[1])
    results_auc = float(results[1].split(": ")[1])

    tolerance = 0.01  # allow for 1/100th of a percent cost difference

    precision_delta_percent = (
        abs(val_precision - results_precision) / val_precision * 100
    )
    precision_similar = (results_precision >= val_precision) or (
        precision_delta_percent <= tolerance
    )

    auc_delta_percent = abs(val_auc - results_auc) / val_precision * 100
    auc_similar = (results_auc >= val_auc) or (auc_delta_percent <= tolerance)

    print(f"Precisiom delta percent: {precision_delta_percent}")
    print(f"AUC delta percent: {auc_delta_percent}")
    print(f"Precision higher/similar: {precision_similar}")
    print(f"AUC higher/similar: {auc_similar}")

    return precision_similar, auc_similar, precision_delta_percent


def verify_supervised_metrics(validation, results):
    (
        precision_similar,
        auc_similar,
        precision_delta_percent,
    ) = compare_supervised_metrics(validation, results)
    assert precision_similar and auc_similar


def verify_sentiment_query(results, validation, query_number, threshold=90):
    if query_number == "18":
        group_cols = ["s_name", "r_date", "sentiment", "sentiment_word"]
    else:
        group_cols = ["item_sk", "sentiment", "sentiment_word"]

    r_grouped = results.groupby(group_cols).size().reset_index()
    s_grouped = validation.groupby(group_cols).size().reset_index()

    t1 = r_grouped
    t2 = s_grouped

    rapids_nrows = t1.shape[0]
    spark_nrows = t2.shape[0]
    res_rows = t1.merge(t2, how="inner", on=list(t1.columns)).shape[0]

    overlap_percent_rapids_denom = res_rows / rapids_nrows * 100
    overlap_percent_spark_denom = res_rows / spark_nrows * 100

    print(
        f"{overlap_percent_rapids_denom}% overlap with {rapids_nrows} rows (RAPIDS denominator)"
    )
    print(
        f"{overlap_percent_spark_denom}% overlap with {spark_nrows} rows (Spark denominator)"
    )

    assert overlap_percent_rapids_denom >= threshold
    assert overlap_percent_spark_denom >= threshold

    return 0


def verify_results(verify_dir):
    """
    verify_dir: Directory which contains verification results
    """
    import cudf
    import dask_cudf
    import cupy as cp
    import dask.dataframe as dd

    QUERY_NUM = get_query_number()

    # Query groupings
    SENTIMENT_QUERIES = (
        "10",
        "18",
        "19",
    )
    CLUSTERING_QUERIES = (
        "20",
        "25",
        "26",
    )
    SUPERVISED_LEARNING_QUERIES = (
        "05",
        "28",
    )

    # Key Thresholds
    SENTIMENT_THRESHOLD = 90

    result_verified = False

    # Short-circuit for the NER query
    if QUERY_NUM in ("27"):
        print("Did not run Correctness check for this query")
        return result_verified

    # Setup validation data
    if QUERY_NUM in SUPERVISED_LEARNING_QUERIES:
        verify_fname = os.path.join(
            verify_dir, f"q{QUERY_NUM}-results/q{QUERY_NUM}-metrics-results.txt"
        )
        result_fname = f"q{QUERY_NUM}-metrics-results.txt"

        with open(verify_fname, "r") as fh:
            validation_data = fh.readlines()

    else:
        result_fname = f"q{QUERY_NUM}-results.parquet/"
        verify_fname = glob.glob(verify_dir + f"q{QUERY_NUM}-results/*.csv")

        validation_data = dd.read_csv(verify_fname, escapechar="\\").compute()

    # Setup results data

    # special case q12 due to the parquet output, which seems to be causing problems
    # for the reader
    # See https://github.com/rapidsai/tpcx-bb-internal/issues/568
    if QUERY_NUM in ("12",):
        results_data = dask_cudf.read_parquet(result_fname + "*.parquet").compute()
        results_data = results_data.to_pandas()

    elif QUERY_NUM in SUPERVISED_LEARNING_QUERIES:
        with open(result_fname, "r") as fh:
            results_data = fh.readlines()

    else:
        results_data = dask_cudf.read_parquet(result_fname).compute()
        results_data = results_data.to_pandas()

    # Verify correctness
    if QUERY_NUM in SUPERVISED_LEARNING_QUERIES:
        print("Supervised Learning Query")
        try:
            verify_supervised_metrics(validation_data, results_data)
            result_verified = True
            print("Correctness Assertion True")
        except AssertionError as error:
            print("Error", error)
            print("Correctness Assertion False")

    elif QUERY_NUM in CLUSTERING_QUERIES:
        print("Clustering Query")
        try:
            cluster_info_validation_path = os.path.join(
                verify_dir, f"q{QUERY_NUM}-results/clustering-results.txt"
            )
            cluster_info_rapids_path = f"q{QUERY_NUM}-results-cluster-info.txt"

            # primary metric
            verify_clustering_query_cost(
                cluster_info_validation_path, cluster_info_rapids_path
            )

            # secondary metric (non-binding)
            verify_clustering_query_labels(validation_data, results_data)

            result_verified = True
            print("Correctness Assertion True")
        except AssertionError as error:
            print("Error", error)
            print("Correctness Assertion False")

    elif QUERY_NUM in SENTIMENT_QUERIES:
        print("Sentiment Analysis Query")
        try:
            verify_sentiment_query(
                results_data, validation_data, QUERY_NUM, threshold=SENTIMENT_THRESHOLD
            )
            result_verified = True
            print("Correctness Assertion True")
        except AssertionError as error:
            print("Error", error)
            print("Correctness Assertion False")

    # scalar results
    elif QUERY_NUM in ("04", "23"):
        print("Scalar Result Query")
        try:
            np.testing.assert_array_almost_equal(
                validation_data.values, results_data.values, decimal=5
            )
            result_verified = True
            print("Correctness Assertion True")
        except AssertionError as error:
            print("Error", error)
            print("Correctness Assertion False")

    else:
        print("Standard ETL Query")
        try:
            assert_dataframes_pseudo_equal(results_data, validation_data)
            result_verified = True
            print("Correctness Assertion True")
        except AssertionError as error:
            print("Error", error)
            print("Correctness Assertion False")

    return result_verified


#################################
# Performance Tracking Automation
#################################


def build_benchmark_googlesheet_payload(config):
    """
    config : dict
    """
    # Don't mutate original dictionary
    data = config.copy()

    # get the hostname of the machine running this workload
    data["hostname"] = socket.gethostname()

    QUERY_NUM = get_query_number()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    query_time = _get_benchmarked_method_time(
        filename="benchmarked_main.csv", query_start_time=config.get("start_time")
    )
    writing_time = _get_benchmarked_method_time(
        filename="benchmarked_write_result.csv",
        query_start_time=config.get("start_time"),
    )
    read_graph_creation_time = _get_benchmarked_method_time(
        filename="benchmarked_read_tables.csv",
        query_start_time=config.get("start_time"),
    )
    if data["get_read_time"] and read_graph_creation_time and query_time:
        ### below contains the computation time
        compute_read_table_time = _get_benchmarked_method_time(
            filename="benchmarked_read_tables.csv",
            field="compute_time_seconds",
            query_start_time=config.get("start_time"),
        )
        # subtracting read calculation time
        query_time = query_time - compute_read_table_time
    else:
        compute_read_table_time = None

    # get library info
    library_info = generate_library_information()
    data.update(library_info)

    import blazingsql
    payload = OrderedDict(
        {
            "Query Number": int(QUERY_NUM),
            "Protocol": data.get("protocol"),
            "NVLINK": data.get("nvlink", "NA"),
            "Infiniband": data.get("infiniband", "NA"),
            "Query Type": "blazing" if is_blazing_query() else "dask",
            "File Format": data.get("file_format"),
            "Time (seconds)": query_time + writing_time
            if query_time and writing_time
            else "NA",
            "Query Time(seconds)": query_time if query_time else "NA",
            "Writing Results Time": writing_time if writing_time else "NA",
            # read time
            "Compute Read + Repartition small table Time(seconds)": compute_read_table_time
            if compute_read_table_time
            else "NA",
            "Graph Creation time(seconds)": read_graph_creation_time
            if read_graph_creation_time
            else "NA",
            "Machine Setup": data.get("hostname"),
            "Data Location": data.get("data_dir"),
            "Repartition_small_table": data.get("repartition_small_table"),
            "Result verified": data.get("result_verified"),
            "Current Time": current_time,
            "Device Memory Limit": data.get("device_memory_limit"),
            "cuDF Version": data.get("cudf"),
            "Dask Version": data.get("dask"),
            "Distributed Version": data.get("distributed"),
            "Dask-CUDA Version": data.get("dask-cuda"),
            "UCX-py Version": data.get("ucx-py"),
            "UCX Version": data.get("ucx"),
            "RMM Version": data.get("rmm"),
            "cuML Version": data.get("cuml"),
            "CuPy Version": data.get("cupy"),
            "Num 16GB workers": data.get("16GB_workers"),
            "Num 32GB workers": data.get("32GB_workers"),
            "Query Status": data.get("query_status", "Unknown"),
            "BlazingSQL version":  blazingsql.__version__  if is_blazing_query() else "",
            "allocator": os.environ.get("BLAZING_ALLOCATOR_MODE", "managed") if is_blazing_query() else "",
            "network_interface": os.environ.get("INTERFACE", "ib0") if is_blazing_query() else "",
            "config_options": str(get_config_options()) if is_blazing_query() else "",
        }
    )
    payload = list(payload.values())
    return payload


def is_blazing_query():
    """
    Method that returns true if `blazingsql` is imported returns false otherwise
    """
    return "blazingsql" in sys.modules

def _get_benchmarked_method_time(
    filename, field="elapsed_time_seconds", query_start_time=None
):
    """
    Returns the `elapsed_time_seconds` field from files generated using the `benchmark` decorator.
    """
    import cudf

    try:
        benchmark_results = cudf.read_csv(filename)
        benchmark_time = benchmark_results[field].iloc[0]
    except FileNotFoundError:
        benchmark_time = None

    return benchmark_time


def generate_library_information():
    KEY_LIBRARIES = [
        "cudf",
        "cuml",
        "dask",
        "distributed",
        "ucx",
        "ucx-py",
        "dask-cuda",
        "rmm",
        "cupy",
    ]

    conda_list_command = (
        os.environ.get("CONDA_PREFIX").partition("envs")[0] + "bin/conda list"
    )
    result = subprocess.run(
        conda_list_command, stdout=subprocess.PIPE, shell=True
    ).stdout.decode("utf-8")
    df = pd.DataFrame(
        [x.split() for x in result.split("\n")[3:]],
        columns=["library", "version", "build", "channel"],
    )
    df = df[df.library.isin(KEY_LIBRARIES)]

    lib_dict = dict(zip(df.library, df.version))
    return lib_dict


def push_payload_to_googlesheet(config):
    if os.environ.get("GOOGLE_SHEETS_CREDENTIALS_PATH", None):
      if not config.get("tab") or not config.get("sheet"):
          print("Must pass a sheet and tab name to use Google Sheets automation")
          return 1

      scope = [
          "https://spreadsheets.google.com/feeds",
          "https://www.googleapis.com/auth/drive",
      ]
      credentials_path = os.environ["GOOGLE_SHEETS_CREDENTIALS_PATH"]

      credentials = ServiceAccountCredentials.from_json_keyfile_name(
          credentials_path, scope
      )
      gc = gspread.authorize(credentials)
      payload = build_benchmark_googlesheet_payload(config)
      s = gc.open(config["sheet"])
      tab = s.worksheet(config["tab"])
      tab.append_row(payload)


#################################
# Query Utilities
#################################


def left_semi_join(df_1, df_2, left_on, right_on):
    """
        Pefrorm left semi join b/w tables
    """
    left_merge = lambda df_1, df_2: df_1.merge(
        df_2, left_on=left_on, right_on=right_on, how="leftsemi"
    )

    ## asserting that number of partitions of the right frame is always 1
    assert df_2.npartitions == 1

    return df_1.map_partitions(left_merge, df_2.to_delayed()[0], meta=df_1._meta)


def convert_datestring_to_days(df):
    import cudf

    df["d_date"] = (
        cudf.to_datetime(df["d_date"], format="%Y-%m-%d")
        .astype("datetime64[s]")
        .astype("int64")
        / 86400
    )
    df["d_date"] = df["d_date"].astype("int64")
    return df


def train_clustering_model(training_df, n_clusters, max_iter, n_init):
    """Trains a KMeans clustering model on the 
    given dataframe and returns the resulting
    labels and WSSSE"""

    from cuml.cluster.kmeans import KMeans

    best_sse = 0
    best_model = None

    # Optimizing by doing multiple seeding iterations.
    for i in range(n_init):
        model = KMeans(
            oversampling_factor=0,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=np.random.randint(0, 500),
            init="k-means++",
        )
        model.fit(training_df)

        score = model.inertia_

        if best_model is None:
            best_sse = score
            best_model = model

        elif abs(score) < abs(best_sse):
            best_sse = score
            best_model = model

    return {
        "cid_labels": best_model.labels_,
        "wssse": best_model.inertia_,
        "cluster_centers": best_model.cluster_centers_,
        "nclusters": n_clusters,
    }
