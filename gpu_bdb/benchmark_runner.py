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


dask_qnums = [str(i).zfill(2) for i in range(1, 31)]
bsql_qnums = [str(i).zfill(2) for i in range(1, 31)]


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster, import_query_libs
    from bdb_tools.utils import run_query, gpubdb_argparser
    from bdb_tools import RMMResourceMonitor
    from bdb_tools import DaskTaskLogger
    import_query_libs()
    config = gpubdb_argparser()
    config["run_id"] = uuid.uuid4().hex
    include_blazing = config.get("benchmark_runner_include_bsql")

    dask_queries = {
        qnum: load_query(qnum, f"queries/q{qnum}/gpu_bdb_query_{qnum}.py")
        for qnum in dask_qnums
    }

    if include_blazing:
        bsql_queries = {
            qnum: load_query(qnum, f"queries/q{qnum}/gpu_bdb_query_{qnum}_sql.py")
            for qnum in bsql_qnums
        }

    client, bc = attach_to_cluster(config, create_blazing_context=include_blazing)
    # Preload required libraries for queries on all workers
    client.run(import_query_libs)

    base_path = os.getcwd()

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

            rmm_analyzer=RMMResourceMonitor(client=client,
                                            outputdir=os.getenv('OUTPUT_DIR', '/tmp'))
            dasktasklog=DaskTaskLogger( client=client,
                                        outputdir=os.getenv('OUTPUT_DIR', '/tmp'))
            #FIXME: OUTPUT_DIR is not managed by gpu-bdb, might want to pick that up into the config
            for r in range(N_REPEATS):
                if config.get('benchmark_runner_log_rmm',False):
                    rmm_analyzer.begin_logging( prefix=f"rmmlog{qnum}")
                if config.get('benchmark_runner_log_tasks',False):
                    dasktasklog.mark_begin()
                run_query(config=config, client=client, query_func=q_func)
                rmm_analyzer.stop_logging()
                dasktasklog.save_tasks( prefix=f"dasktasklog{qnum}")
                client.run(gc.collect)
                client.run_on_scheduler(gc.collect)
                gc.collect()
                time.sleep(3)
