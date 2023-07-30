from bdb_tools.utils import benchmark, gpubdb_argparser, run_query
from bdb_tools.readers import build_reader
from cudf.core.dtypes import Decimal64Dtype
import os, subprocess, math, time


config = gpubdb_argparser()

spark_schema_dir = f"{os.getcwd()}/spark_table_schemas/"

# these tables have extra data produced by bigbench dataGen
refresh_tables = [
    "customer",
    "customer_address",
    "inventory",
    "item",
    "item_marketprices",
    "product_reviews",
    "store_returns",
    "store_sales",
    "web_clickstreams",
    "web_returns",
    "web_sales",
]
tables = [table.split(".")[0] for table in os.listdir(spark_schema_dir)]

scale = [x for x in config["data_dir"].split("/") if "sf" in x][0]
part_size = 2
chunksize = "128 MiB"

# Spark uses different names for column types, and RAPIDS doesn't yet support Decimal types.
def get_schema(table):
    with open(f"{spark_schema_dir}{table}.schema") as fp:
        schema = fp.read()
        names = [line.replace(",", "").split()[0] for line in schema.split("\n")]
        types = [
            line.replace(", ", "")
            .split()[1]
            .replace("bigint", "int")
            .replace("string", "str")
            for line in schema.split("\n")
        ]

        return names, types


def read_csv_table(table, chunksize="256 MiB"):
    # build dict of dtypes to use when reading CSV
    names, types = get_schema(table)
    types_dec_as_str = ['str' if 'decimal' in t else t for t in types]
    dtype_dec_as_str = {names[i]: types_dec_as_str[i] for i in range(0, len(names))}

    data_dir = config["data_dir"].split('parquet_')[0]
    base_path = f"{data_dir}/data/{table}"
    files = os.listdir(base_path)
    # item_marketprices has "audit" files that should be excluded
    if table == "item_marketprices":
        paths = [
            f"{base_path}/{fn}"
            for fn in files
            if "audit" not in fn and os.path.getsize(f"{base_path}/{fn}") > 0
        ]
        base_path = f"{data_dir}/data_refresh/{table}"
        paths = paths + [
            f"{base_path}/{fn}"
            for fn in os.listdir(base_path)
            if "audit" not in fn and os.path.getsize(f"{base_path}/{fn}") > 0
        ]
        df = dask_cudf.read_csv(
            paths, sep="|", names=names, dtype=dtype_dec_as_str, chunksize=chunksize, quoting=3
        )
    else:
        paths = [
            f"{base_path}/{fn}"
            for fn in files
            if os.path.getsize(f"{base_path}/{fn}") > 0
        ]
        if table in refresh_tables:
            base_path = f"{data_dir}/data_refresh/{table}"
            paths = paths + [
                f"{base_path}/{fn}"
                for fn in os.listdir(base_path)
                if os.path.getsize(f"{base_path}/{fn}") > 0
            ]
        df = dask_cudf.read_csv(
            paths, sep="|", names=names, dtype=types_dec_as_str, chunksize=chunksize, quoting=3
        )

    for i, dtype in enumerate(types):
        if 'decimal' in dtype:
            precision = int(dtype[dtype.find("(")+1:dtype.find(",")])
            scale = int(dtype[dtype.find(",")+1:dtype.find(")")])
            df[names[i]] = df[names[i]].astype(Decimal64Dtype(precision, scale))

    return df


def multiplier(unit):
    if unit == "G":
        return 1
    elif unit == "T":
        return 1000
    else:
        return 0


# we use size of the CSV data on disk to determine number of Parquet partitions
def get_size_gb(table):
    data_dir = config["data_dir"].split('parquet_')[0]
    path = data_dir + "/data/" + table
    size = subprocess.check_output(["du", "-sh", path]).split()[0].decode("utf-8")
    unit = size[-1]

    size = math.ceil(float(size[:-1])) * multiplier(unit)

    if table in refresh_tables:
        path = data_dir + "/data_refresh/" + table
        refresh_size = (
            subprocess.check_output(["du", "-sh", path]).split()[0].decode("utf-8")
        )
        size = size + math.ceil(float(refresh_size[:-1])) * multiplier(refresh_size[-1])

    return size


def repartition(table, outdir, npartitions=None, chunksize=None, compression="snappy"):
    size = get_size_gb(table)
    if npartitions is None:
        npartitions = max(1, size)

    print(
        f"Converting {table} of {size} GB to {npartitions} parquet files, chunksize: {chunksize}"
    )
    read_csv_table(table, chunksize).repartition(
        npartitions=npartitions
    ).to_parquet(outdir + table, compression=compression, index=False)


def main(client, config):
    # location you want to write Parquet versions of the table data
    data_dir = config["data_dir"].split('parquet_')[0]
    outdir = f"{data_dir}/parquet_{part_size}gb_decimal/"

    t0 = time.time()
    for table in tables:
        size_gb = get_size_gb(table)
        # product_reviews has lengthy strings which exceed cudf's max number of characters per column
        # we use smaller partitions to avoid overflowing this character limit
        if table == "product_reviews":
            npartitions = max(1, int(size_gb / 1))
        else:
            npartitions = max(1, int(size_gb / part_size))
        repartition(table, outdir, npartitions, chunksize, compression="snappy")
    print(f"Load test with chunk size of {chunksize} took {time.time() - t0:.2f}s")
    return cudf.DataFrame()


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
