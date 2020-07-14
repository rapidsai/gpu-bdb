import glob
import re
import os
import gc
import time


os.environ["tpcxbb_benchmark_sweep_run"] = "True"

N_REPEATS = 2


def get_qnum_from_filename(name):
    m = re.search("[0-9]{2}", name).group()
    return m


dask_qnums = [str(i).zfill(2) for i in range(1, 31)]

# Not all queries are implemented with BSQL
bsql_query_files = sorted(glob.glob("./queries/q*/t*_sql.py"))
bsql_qnums = [get_qnum_from_filename(x.split("/")[-1]) for x in bsql_query_files]

dask_qnums = [
    "01",
    #     "07",
    #     "09",
]

bsql_qnums = [
    "01",
    #     "07",
    #     "09",
]

if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster, import_query_libs
    from xbb_tools.utils import run_query, tpcxbb_argparser
    import importlib

    import_query_libs()

    dask_queries = {
        qnum: importlib.import_module(f"queries.q{qnum}.tpcx_bb_query_{qnum}").main
        for qnum in dask_qnums
    }

    bsql_queries = {
        qnum: importlib.import_module(f"queries.q{qnum}.tpcx_bb_query_{qnum}_sql").main
        for qnum in bsql_qnums
    }

    cli_args = tpcxbb_argparser()
    client, bc = attach_to_cluster(cli_args, create_blazing_context=True)

    # Preload required libraries for queries on all workers
    client.run(import_query_libs)

    base_path = os.getcwd()

    # Run Pure Dask Queries
    print("Pure Dask Queries")
    for qnum, q_func in dask_queries.items():
        print(qnum)

        qpath = f"{base_path}/queries/q{qnum}/"
        os.chdir(qpath)
        if os.path.exists("current_query_num.txt"):
            os.remove("current_query_num.txt")
        with open("current_query_num.txt", "w") as fp:
            fp.write(qnum)

        for r in range(N_REPEATS):
            run_query(cli_args=cli_args, client=client, query_func=q_func)
            client.run(gc.collect)
            client.run_on_scheduler(gc.collect)
            gc.collect()
            time.sleep(3)

    # Run BSQL Queries
    print("Blazing Queries")
    for qnum, q_func in bsql_queries.items():
        print(qnum)

        qpath = f"{base_path}/queries/q{qnum}/"
        os.chdir(qpath)
        if os.path.exists("current_query_num.txt"):
            os.remove("current_query_num.txt")
        with open("current_query_num.txt", "w") as fp:
            fp.write(qnum)

        for r in range(N_REPEATS):
            run_query(
                cli_args=cli_args, client=client, query_func=q_func, blazing_context=bc
            )
            client.run(gc.collect)
            client.run_on_scheduler(gc.collect)
            gc.collect()
            time.sleep(3)
