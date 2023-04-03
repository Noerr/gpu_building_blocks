# Makefile to capture the different build variants

HIP_KERNEL_SRC	= microapps/kernel_parts.hip.cpp
CORE_DEVICE_SRC = accelerator_api/core_device_API_hip.cpp
DV_SRC = distributed_variable/distributed_variable.cpp
#SRC	 = src/main.c
#OBJ	 = $(patsubst src%,obj$(VERSION)%,$(patsubst %.c,%.o,$(SRC)))

CXX_FLAGS = -std=c++17
CC = CC
HIPCC = hipcc

# NVIDIA/CUDA HIP Platform:
INSTR_LIB_LINK = -lnvToolsExt
RTC_LINK = -lnvrtc

all:	microapp.mpi_hip_compile_time.exe microapp.mpi_hip_runtime_module.exe microapp.mpi_hip_rtc.exe

#
# For each application just call make with VERSION set correctly
microapp.%:  dir.%
	$(MAKE) VERSION=$* app$*.elf

#
# Build each applications objects into a seprate directory
# So we need to make sure the obj directory exists.
dir.%:
	@- if ! test -e dir.$*; then mkdir dir.$*; fi

microapp.mpi_hip_compile_time.exe:   dir.mpi_hip_compile_time $(HIP_KERNEL_SRC) $(CORE_DEVICE_SRC) $(DV_SRC) microapps/prototype_runtime_load_driver_with_MPI.cpp
	$(eval DVARIANT=-DKERNEL_LINK_METHOD_COMPILE_TIME)
	$(HIPCC) $(DVARIANT) -I./accelerator_api/  -c  $(HIP_KERNEL_SRC) -o ./$</kernel_parts_hip.o 
	$(HIPCC) $(DVARIANT) $(CXX_FLAGS)  -I./accelerator_api/  -c  $(CORE_DEVICE_SRC) -o ./$</core_device_API_hip.o
	$(CC) $(DVARIANT) $(CXX_FLAGS) -I./accelerator_api -I./distributed_variable -c $(DV_SRC) -o ./$</dist_var.o
	$(CC) $(DVARIANT) $(CXX_FLAGS) -I./accelerator_api -I./distributed_variable -c microapps/prototype_runtime_load_driver_with_MPI.cpp -o ./$</driver.o
	
	$(CC) $(CXX_FLAGS) ./$</kernel_parts_hip.o ./$</core_device_API_hip.o ./$</dist_var.o ./$</driver.o  $(INSTR_LIB_LINK) -o $@


microapp.mpi_hip_rtc.exe:   dir.mpi_hip_rtc $(HIP_KERNEL_SRC) $(CORE_DEVICE_SRC) $(DV_SRC) microapps/prototype_runtime_load_driver_with_MPI.cpp
	$(eval DVARIANT=-DKERNEL_LINK_METHOD_RTC)
	$(HIPCC) $(DVARIANT) -I./accelerator_api/  -c  $(HIP_KERNEL_SRC) -o ./$</kernel_parts_hip.o 
	$(HIPCC) $(DVARIANT) $(CXX_FLAGS)  -I./accelerator_api/  -c  $(CORE_DEVICE_SRC) -o ./$</core_device_API_hip.o
	$(CC) $(DVARIANT) $(CXX_FLAGS) -I./accelerator_api -I./distributed_variable -c $(DV_SRC) -o ./$</dist_var.o
	$(CC) $(DVARIANT) $(CXX_FLAGS) -I./accelerator_api -I./distributed_variable -c microapps/prototype_runtime_load_driver_with_MPI.cpp -o ./$</driver.o
	
	$(CC) $(CXX_FLAGS) ./$</kernel_parts_hip.o ./$</core_device_API_hip.o ./$</dist_var.o ./$</driver.o  $(INSTR_LIB_LINK) $(RTC_LINK) -o $@


microapp.mpi_hip_runtime_module.exe:   dir.mpi_hip_runtime_module $(HIP_KERNEL_SRC) $(CORE_DEVICE_SRC) $(DV_SRC) microapps/prototype_runtime_load_driver_with_MPI.cpp
	$(eval DVARIANT=-DKERNEL_LINK_METHOD_RUNTIME_MODULE)
	$(HIPCC) --shared $(DVARIANT) -I./accelerator_api/  -c  $(HIP_KERNEL_SRC) -fPIC -o ./$</hip_kernel_module.so 
	$(HIPCC) $(DVARIANT) $(CXX_FLAGS)  -I./accelerator_api/  -c  $(CORE_DEVICE_SRC) -o ./$</core_device_API_hip.o
	$(CC) $(DVARIANT) $(CXX_FLAGS) -I./accelerator_api -I./distributed_variable -c $(DV_SRC) -o ./$</dist_var.o
	$(CC) $(DVARIANT) $(CXX_FLAGS) -I./accelerator_api -I./distributed_variable -c microapps/prototype_runtime_load_driver_with_MPI.cpp -o ./$</driver.o
	
	$(CC) $(CXX_FLAGS)                         ./$</core_device_API_hip.o ./$</dist_var.o ./$</driver.o  $(INSTR_LIB_LINK) -o $@


