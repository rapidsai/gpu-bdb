import glob
import re
import os
import gc
import time

N_REPEATS = 1


def get_qnum_from_filename(name):
    m = re.search("[0-9]{2}", name).group()
    return m


dask_qnums = [str(i).zfill(2) for i in range(1, 31)]
# Not all queries are implemented with BSQL
bsql_query_files = sorted(glob.glob("./queries/q*/t*_sql.py"))
bsql_qnums = [get_qnum_from_filename(x.split("/")[-1]) for x in bsql_query_files]

def load_query(qnum, fn):
    import importlib, types
    loader = importlib.machinery.SourceFileLoader(qnum, fn)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod.main

if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster, import_query_libs
    from xbb_tools.utils import run_query, tpcxbb_argparser

    import_query_libs()
    dask_queries = {
        qnum: load_query(qnum, f"queries/q{qnum}/tpcx_bb_query_{qnum}.py")
        for qnum in dask_qnums
    }

    bsql_queries = {
        qnum: load_query(qnum, f"queries/q{qnum}/tpcx_bb_query_{qnum}_sql.py")
        for qnum in bsql_qnums
    }

    
    config = tpcxbb_argparser()
    include_blazing = config.get("benchmark_runner_include_bsql")
    client, bc = attach_to_cluster(config, create_blazing_context=include_blazing)

    # Preload required libraries for queries on all workers
    client.run(import_query_libs)

    base_path = os.getcwd()

    # Run Pure Dask Queries
    if len(dask_qnums) > 0:
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
                run_query(config=config, client=client, query_func=q_func)
                client.run(gc.collect)
                client.run_on_scheduler(gc.collect)
                gc.collect()
                time.sleep(3)

    # Run BSQL Queries
    if include_blazing and len(bsql_qnums) > 0:
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
                    config=config,
                    client=client,
                    query_func=q_func,
                    blazing_context=bc,
                )
                client.run(gc.collect)
                client.run_on_scheduler(gc.collect)
                gc.collect()
                time.sleep(3)
