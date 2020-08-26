Setup:
1. Somewhere you have Docker permissions, build image:
```bash
cd $TPCX_BB/tpcx_bb/benchmark_runner/draco
docker build -t draco .
# update image id
docker tag bb38976d03cf dockerhub_username/draco:8_24_0
docker push dockerhub_username/draco
```

2. In Draco, create dask-local-directory:
```bash
mkdir -p ~/dask-local-directory
```

3. Update bash vars `run.sh` to make sure you're specifying the right image and number of workers
4. Run:
```bash
bash benchmark_runner/draco/run.sh
```

Notes:
Once your scheduler node is running, the client will [wait](../wait.py) for the expected number of workers to connect before starting the load test and running the queries.

If you're on the VPN, you can load the Dask dashboard at the IP address in `~/dask-local-directory/scheduler.json`.
