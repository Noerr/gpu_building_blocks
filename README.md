# GPU Building Blocks - key techonolgy tests and demonstration
Prototypes and Functionality tests for GPU &amp; MPI programming models relevant to plasma modeling code
Uses HIP programming model to target both NVIDIA and AMD GPUs and GPU pointers with MPI One-sided methods.




## Build the microapp test applications

Note:  These commands have been tested on NERSC Perlmutter's default module environment except as follows:
```
module load hip
```

### Executable driver that prototypes runtime loading and selection of GPU kernel to apply
```
gpu_building_blocks/microapps> CC -I../accelerator_api -I../distributed_variable -o cuda_runtime_kernel_link_prototype.exe  ../accelerator_api/core_device_API_cuda.cpp  prototype_runtime_load_driver.cpp -ldl -lcudart -lnvToolsExt

```

### Executable driver that additionally adds MPI one-side "PUT" messaging mimicing an unstructured ghost face exchange 

The runtime load shared library module approach:
```
CC  -std=c++17 -I../accelerator_api -I../distributed_variable -o cuda_runtime_kernel_link_prototype_wMPI.exe  ../accelerator_api/core_device_API_cuda.cpp ../distributed_variable/distributed_variable.cpp  prototype_runtime_load_driver_with_MPI.cpp -ldl -lcudart -lnvToolsExt
```

More typical mixed-source kernel and host code compile time approach:
```
CC -DKERNEL_LINK_METHOD_COMPILE_TIME -std=c++17 -I../accelerator_api -I../distributed_variable -o cuda_compile_time_prototype_wMPI.exe  ../accelerator_api/core_device_API_cuda.cpp ../distributed_variable/distributed_variable.cpp  prototype_runtime_load_driver_with_MPI.cpp  cuda_kernel_parts.cu  -lcudart -lnvToolsExt
```

Runtime comilation of kernels from source code strings:
```
CC -DKERNEL_LINK_METHOD_RTC  -std=c++17 -I../accelerator_api -I../distributed_variable -o cuda_RTC_prototype_wMPI.exe  ../accelerator_api/core_device_API_cuda.cpp ../distributed_variable/distributed_variable.cpp  prototype_runtime_load_driver_with_MPI.cpp  cuda_kernel_parts.cu  -lcudart -lnvToolsExt -lnvrtc
```


### An example kernel library for runtime works with either prototype driver microapp
```
gpu_building_blocks/microapps> nvcc -shared  -arch=compute_80 -I../accelerator_api -o libkernel_parts.so cuda_kernel_parts.cu --compiler-options '-fPIC'
```


## Run the microapp test

### Basic driver

```
gpu_building_blocks/microapps> ./cuda_runtime_kernel_link_prototype.exe 6 ./libkernel_parts.so initialize_element_kernel

initialize_element_kernel:  0x148159ffcc63 ,  0x148159ffcc63
launching fn by ptr 0x148159ffcc63
launching fn by ptr 0x148159ffcdcc
P 0: 1000, 1001, 1002, 1003, 1004, 1005

```


### With MPI distributed variable

```
export MPICH_GPU_SUPPORT_ENABLED=1
srun  --gpu-bind=none --time-min=2 -n16 --ntasks-per-node=4 -C gpu --gpus-per-task=1 ./cuda_runtime_kernel_link_prototype_wMPI.exe 46656 ./libkernel_parts.so initialize_element_kernel
```

```
Process Grid Dims: 4, 2, 2
Cube Element Grid Dims (per process): 36, 36, 36 [though something more like cube faces are exchanged].
Completed 20 ghost face exchange iterations.
    First epoch: 76433 microseconds
    Remaining epochs: 1232 microseconds,  64.8421 per iteration.
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


