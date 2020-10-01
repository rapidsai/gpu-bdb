#!/bin/bash

ACCOUNT=rapids
PARTITION=luna
WORKERS=15

#IMAGE=nickb500/selene-tpcxbb:2020_09_17
#IMAGE=nickb500/selene-tpcxbb-small:2020_09_17
#IMAGE=gitlab-master.nvidia.com/rapids/workflows/tpcx-bb:latest
#IMAGE=gitlab-master.nvidia.com/rapids/workflows/tpcx-bb:2020_09_28
IMAGE=/lustre/fsw/rapids/tpcx-bb/containers/tpcx-bb-20200928.sqsh

DATA_PATH="/lustre/fsw/rapids"
MOUNT_PATH="/tpcx-bb-data/"
TPCX_BB_HOME="$HOME/tpcx-bb"

srun \
	--account $ACCOUNT \
	--partition $PARTITION \
	--nodes $WORKERS \
	--container-mounts $DATA_PATH:$MOUNT_PATH,$HOME:$HOME \
	--container-image $IMAGE \
	bash -c "$TPCX_BB_HOME/tpcx_bb/cluster_configuration/spawn-workers.sh"
