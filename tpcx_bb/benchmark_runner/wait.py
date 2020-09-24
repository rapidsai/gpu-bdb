from dask.distributed import Client
import sys, time, yaml

config_file = sys.argv[1]
with open(config_file) as fp:
    conf = yaml.safe_load(fp)

# handle arbitrary sized worker
for key in conf.keys():
    if '_workers' in key:
        expected_workers = int(conf[key])

ready = False
while not ready:
    with Client(scheduler_file=conf['scheduler_file']) as client:
        workers = client.scheduler_info()['workers']
        if len(workers) < expected_workers:
            print(f'Expected {expected_workers} but got {len(workers)}, waiting..')
            time.sleep(10)
        else:
            print(f'Got all {len(workers)} workers')
            ready = True
