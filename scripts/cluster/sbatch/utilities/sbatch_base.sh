#!/bin/bash


export SLURM_PREFIX="./scripts/cluster/slurm"

if hostname | grep -qi "discovery"; then
    # the usc carc discovery env
    export CLUSTER_NAME="discovery"

    # may change the following two according to project details
    export PROJECT_ACCOUNT="neiswang_1391"
    export PROJECT_PREFIX="/project/${PROJECT_ACCOUNT}"
    export HOME_PREFIX="/home1/$USER"
    export SCRATCH_PREFIX="/scratch1/$USER"
else
    # the ucsd access expanse env
    export CLUSTER_NAME="expanse"

    # may change the following two according to project details
    export PROJECT_ACCOUNT="mia346"
    export PROJECT_PREFIX="/expanse/lustre/projects/${PROJECT_ACCOUNT}/$USER"
    export HOME_PREFIX="/home/$USER"
    export SCRATCH_PREFIX="/expanse/lustre/scratch/$USER/temp_project"
fi
