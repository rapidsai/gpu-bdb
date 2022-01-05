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
sql_qnums = [str(i).zfill(2) for i in range(1, 31)]

for query in exclude:
  sql_qnums = [q for q in sql_qnums if q != query]
print(sql_qnums)


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

    include_sql = True
    if include_sql:
        sql_queries = {
            qnum: load_query(qnum, f"queries/q{qnum}/gpu_bdb_query_{qnum}_dask_sql.py")
            for qnum in sql_qnums
        }

    client, c = attach_to_cluster(config, create_sql_context=include_sql)
    # Preload required libraries for queries on all workers
    client.run(import_query_libs)

    base_path = os.getcwd()

    # Run Dask SQL Queries
    if include_sql and len(sql_qnums) > 0:
        print("Dask SQL Queries")
        for r in range(N_REPEATS):
          for qnum, q_func in sql_queries.items():
                print(f"{r}: {qnum}")

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
