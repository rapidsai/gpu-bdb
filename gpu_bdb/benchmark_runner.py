import glob
import re
import os
import gc
import time
import uuid

N_REPEATS = 5


def get_qnum_from_filename(name):
    m = re.search("[0-9]{2}", name).group()
    return m


def load_query(qnum, fn):
    import importlib, types
    loader = importlib.machinery.SourceFileLoader(qnum, fn)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod.main


dask_qnums = [str(i).zfill(2) for i in map(int,os.getenv("DASK_QNUMS"," ".join(map(str,range(1, 31)))).split())]
sql_qnums = [str(i).zfill(2) for i in map(int,os.getenv("BSQL_QNUMS"," ".join(map(str,range(1, 31)))).split())]

from random import shuffle
shuffle(dask_qnums)

if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster, import_query_libs
    from bdb_tools.utils import run_query, gpubdb_argparser

    import_query_libs()
    config = gpubdb_argparser()
    config["run_id"] = uuid.uuid4().hex

    include_sql = config.get("benchmark_runner_include_sql")

    dask_queries = {
        qnum: load_query(qnum, f"queries/q{qnum}/gpu_bdb_query_{qnum}.py")
        for qnum in dask_qnums
    }

    if include_sql:
        sql_queries = {
            qnum: load_query(qnum, f"queries/q{qnum}/gpu_bdb_query_{qnum}_dask_sql.py")
            for qnum in sql_qnums
        }
    else:
        dask_queries = {
            qnum: load_query(qnum, f"queries/q{qnum}/gpu_bdb_query_{qnum}.py")
            for qnum in dask_qnums
        }

    client, c = attach_to_cluster(config, create_sql_context=include_sql)
    # Preload required libraries for queries on all workers
    client.run(import_query_libs)

    base_path = os.getcwd()

    if config.get('benchmark_runner_log_rmm', False) or config.get('benchmark_runner_log_tasks', False):

        from bdb_tools import RMMResourceMonitor
        from bdb_tools import DaskTaskLogger

        rmm_analyzer=RMMResourceMonitor(client=client,
                                        outputdir=os.getenv('OUTPUT_DIR', '/tmp'))
        dasktasklog=DaskTaskLogger( client=client,
                                    outputdir=os.getenv('OUTPUT_DIR', '/tmp'))

        orig_run_query=run_query
        def logged_run_query( *args, **kwargs ):
            rmm_analyzer.begin_logging( prefix=f"rmmlog{qnum}")
            dasktasklog.mark_begin()
            orig_run_query( *args, **kwargs )
            rmm_analyzer.stop_logging()
            dasktasklog.save_tasks( prefix=f"dasktasklog{qnum}")

        run_query=logged_run_query

    # Run Dask SQL Queries
    if include_sql and len(sql_qnums) > 0:
        print("Dask SQL Queries")
        for r in range(N_REPEATS):
          for qnum, q_func in sql_queries.items():
                print(f"run {r+1}: q{qnum}")

                qpath = f"{base_path}/queries/q{qnum}/"
                os.chdir(qpath)
                if os.path.exists("current_query_num.txt"):
                    os.remove("current_query_num.txt")
                with open("current_query_num.txt", "w") as fp:
                    fp.write(qnum)

                    run_query(
                        config=config,
                        client=client,
                        query_func=q_func,
                        sql_context=c,
                    )
                    client.run(gc.collect)
                    client.run_on_scheduler(gc.collect)
                    gc.collect()
                    time.sleep(3)

    # Run Pure Dask Queries
    if len(dask_qnums) > 0:
        print("Pure Dask Queries")
        for r in range(N_REPEATS):
            for qnum, q_func in dask_queries.items():
                print(f"run {r+1}: q{qnum}")

                qpath = f"{base_path}/queries/q{qnum}/"
                os.chdir(qpath)
                if os.path.exists("current_query_num.txt"):
                    os.remove("current_query_num.txt")
                with open("current_query_num.txt", "w") as fp:
                    fp.write(qnum)

                    run_query(config=config, client=client, query_func=q_func)
                    client.run(gc.collect)
                    client.run_on_scheduler(gc.collect)
                    gc.collect()
                    time.sleep(3)


