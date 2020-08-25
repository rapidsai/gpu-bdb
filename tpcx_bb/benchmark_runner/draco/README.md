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
