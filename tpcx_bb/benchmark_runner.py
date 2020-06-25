import os

os.environ["tpcxbb_benchmark_sweep_run"] = "True"
import time

qnums = [str(i).zfill(2) for i in range(1, 31)]
if __name__ == "__main__":
    from xbb_tools.cluster_startup import maybe_start_cluster, import_query_libs
    from xbb_tools.utils import run_dask_cudf_query, tpcxbb_argparser
    import importlib

    import_query_libs()

    q_func_d = {
        qnum: importlib.import_module(
            f"queries.q{qnum}.tpcx_bb_query_{qnum}"
        ).main
        for qnum in qnums
    }

    cli_args = tpcxbb_argparser()
    cluster, client = maybe_start_cluster(cli_args)

    # Preload required libraries for queries on all workers
    client.run(import_query_libs)

    base_path = os.getcwd()
    for qnum, q_func in q_func_d.items():
        qpath = f"{base_path}/queries/q{qnum}/"
        os.chdir(qpath)
        if os.path.exists("current_query_num.txt"):
            os.remove("current_query_num.txt")
        with open("current_query_num.txt", "w") as fp:
            fp.write(qnum)

        run_dask_cudf_query(
            cli_args=cli_args, cluster=cluster, client=client, query_func=q_func
        )
        time.sleep(3)
