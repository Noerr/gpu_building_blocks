# GPU Building Blocks - key techonolgy tests and demonstration
Prototypes and Functionality tests for GPU &amp; MPI programming models relevant to plasma modeling code
Uses HIP programming model to target both NVIDIA and AMD GPUs and GPU pointers with MPI One-sided methods.




## Build the microapp test applications

Note:  These commands have been tested on NERSC Perlmutter's default module environment except as follows:
```
module load hip
make
```

Note:  These commands have been tested on OLCF Frontier's default module environment except as follows:
```
module load PrgEnv-amd
module load craype-accel-amd-gfx90a
make
```

The `Makefile` builds three variants of the microapp driver that uses MPI one-side "PUT" messaging mimicing an unstructured ghost face exchange 

- Canonical same-source host and kernel code compiled at same time
- The runtime load shared library module approach
- Runtime comilation of kernels from source code strings




## Run the microapp test with MPI distributed variable

```
export MPICH_GPU_SUPPORT_ENABLED=1
srun  --gpu-bind=none --time-min=2 -n16 --ntasks-per-node=4 -C gpu --gpus-per-task=1 ./microapp.mpi_hip_compile_time.exe <numElements>  initialize_element_kernel


srun  --gpu-bind=none --time-min=2 -n16 --ntasks-per-node=4 -C gpu --gpus-per-task=1 ./microapp.mpi_hip_rtc.exe <numElements>  ./libkernel_parts.so initialize_element_kernel

#note the runtime shared-library module version requires additional argument for the shared library path.  Absolute path may be best.
srun  --gpu-bind=none --time-min=2 -n16 --ntasks-per-node=4 -C gpu --gpus-per-task=1 ./microapp.mpi_hip_runtime_module.exe <numElements>  ./dir.mpi_hip_runtime_module/hip_kernel_module.so  initialize_element_kernel

```

NOTE that on perlmutter, a bug in cray-mpich through at least version 8.1.24 requires disabling GPU binding `--gpu-bind=none` or you'll see an error signature like the following.

```
MPICH ERROR [Rank 1] [job id 6040415.0] [Mon Mar 13 08:52:40 2023] [nid004013] - Abort(673826050) (rank 1 in comm 0): Fatal error in PMPI_Put: Invalid count, error stack:
PMPI_Put(188).........................: MPI_Put(origin_addr=0x1513c3200020, origin_count=2, MPI_DOUBLE, target_rank=0, target_disp=4, target_count=2, MPI_DOUBLE, win=0xa0000000) failed
MPID_Put(994).........................: 
MPIDI_put_safe(677)...................: 
MPIDI_put_unsafe(70)..................: 
MPIDI_POSIX_do_put(142)...............: 
MPIDI_POSIX_cray_get_target_vaddr(206): 
(unknown)(): Invalid count
```


