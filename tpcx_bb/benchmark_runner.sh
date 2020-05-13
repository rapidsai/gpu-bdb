#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION.

# Benchmarking runner shell script

# Example Usage
# ./runner.sh --data_dir=new/data/dir --dask_dir=/dask/dir --file_format=parquet --spark_schema_dir=schema/dir --pool_mode=True --pool_size='1kB'
# Args that are not supplied will use the default.
# The first argument differentiates between which version of the queries to run (blazing vs dask_cudf) implementations,


ARGS=${*:2}

if [ $1 = "blazing" ]; then
    file_pat=*sql.py
elif [ $1 = "dask" ]; then
    file_pat=*[^sql].py
else
    echo "First argument must specify query type (blazing or dask)"
    exit
fi

# Run all the queries
cd queries/
mkdir -p query-tracebacks

for f in q*; do
    echo "Processing $f";
    cd $f;
        for p in $file_pat; do
            QUERY_TAG=$(echo $p | cut -d'.' -f 1 | cut -d'-' -f 3-4)
            TRACEBACK_FNAME="../query-tracebacks/$QUERY_TAG.traceback"
            echo "Query CLI Arguments:\n$ARGS\n\n">$TRACEBACK_FNAME
            echo "Running query $p";
            python $p $ARGS>>$TRACEBACK_FNAME 2>&1
        done
    cd ../
    echo "Sleeping for 3 seconds."
    sleep 3
done
