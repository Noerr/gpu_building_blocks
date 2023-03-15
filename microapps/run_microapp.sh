#!/usr/bin/env bash
#
#
#SBATCH --gpu-bind=none
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH -A m3607_g
#

NUM_ELEMENTS_PER_PE=46656
KERNEL_MODULE_FILENAME=./libkernel_parts.so
KERNEL_REGISTERED_NAME=initialize_element_kernel

EXE=./cuda_runtime_kernel_link_prototype_wMPI.exe

## Alternative RTC approach:
# EXE=./cuda_RTC_prototype_wMPI.exe #./cuda_runtime_kernel_link_prototype_wMPI.exe
# unset KERNEL_MODULE_FILENAME # for RTC and compile-time versions

NTASKS=$(expr $SLURM_JOB_NUM_NODES '*' $SLURM_NTASKS_PER_NODE )

cd ~/gpu_building_blocks/microapps


## Instrumentation / Debugger:
# LAUNCH_PREFIX=ncu

export MPICH_GPU_SUPPORT_ENABLED=1 #use of gpu pointers as buffer args would fail without this.

srun --ntasks-per-node 1 dcgmi profile --pause

srun -n ${NTASKS} ${LAUNCH_PREFIX} ${EXE} ${NUM_ELEMENTS_PER_PE}  ${KERNEL_MODULE_FILENAME} ${KERNEL_REGISTERED_NAME}

srun --ntasks-per-node 1 dcgmi profile --resume

