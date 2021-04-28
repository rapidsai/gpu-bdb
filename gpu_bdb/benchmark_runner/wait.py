import os
import sys
import time
import yaml

from dask.distributed import Client


config_file = sys.argv[1]
with open(config_file) as fp:
    conf = yaml.safe_load(fp)

expected_workers = int(os.environ.get("NUM_WORKERS", 16))

ready = False
while not ready:
    with Client(scheduler_file=conf['scheduler_file_path']) as client:
        workers = client.scheduler_info()['workers']
        if len(workers) < expected_workers:
            print(f'Expected {expected_workers} but got {len(workers)}, waiting..')
            time.sleep(10)
        else:
            print(f'Got all {len(workers)} workers')
            ready = True
