# GPU Building Blocks - key techonolgy tests and demonstration
Prototypes and Functionality tests for GPU &amp; MPI programming models relevant to plasma modeling code




## Build the microapp test applications


### Executable driver that prototypes runtime loading and selection of GPU kernel to apply
```
gpu_building_blocks/microapps> CC -I../accelerator_api -I../distributed_variable -o cuda_runtime_kernel_link_prototype.exe  ../accelerator_api/core_device_API_cuda.cpp  prototype_runtime_load_driver.cpp

```


### An example kernel library for runtime
```
gpu_building_blocks/microapps> nvcc -shared  -arch=compute_80 -I../accelerator_api -o libkernel_parts.so cuda_kernel_parts.cu --compiler-options '-fPIC'
```


## Run the microapp test

```
gpu_building_blocks/microapps> ./cuda_runtime_kernel_link_prototype.exe 6 ./libkernel_parts.so initialize_element_kernel

initialize_element_kernel:  0x148159ffcc63 ,  0x148159ffcc63
launching fn by ptr 0x148159ffcc63
launching fn by ptr 0x148159ffcdcc
P 0: 1000, 1001, 1002, 1003, 1004, 1005

```

