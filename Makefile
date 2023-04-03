# Makefile to capture the different build variants

HIP_KERNEL_SRC	= microapps/kernel_parts.cpp
#SRC	 = src/main.c
#OBJ	 = $(patsubst src%,obj$(VERSION)%,$(patsubst %.c,%.o,$(SRC)))

C_FLAGS = 
CC = CC
HIPCC = hipcc

#ifndef KERNEL_LINK_METHOD_RUNTIME_MODULE
#ifndef KERNEL_LINK_METHOD_COMPILE_TIME
#ifndef KERNEL_LINK_METHOD_RTC

all:	microapp.mpi_hip_compile_time.exe #microapp.mpi_hip_runtime_module.exe microapp.mpi_hip_rtc.exe

#
# For each application just call make with VERSION set correctly
microapp.%:  dir.%
	$(MAKE) VERSION=$* app$*.elf

#
# Build each applications objects into a seprate directory
# So we need to make sure the obj directory exists.
dir.%:
	@- if ! test -e dir.$*; then mkdir dir.$*; fi

microapp.mpi_hip_compile_time.exe:   dir.mpi_hip_compile_time
	$(HIPCC)  -I./accelerator_api/  -c  $(HIP_KERNEL_SRC) -o ./dir.mpi_hip_compile_time/$(HIP_KERNEL_SRC).o #-o$@ $(C_FLAGS) $(OBJ)


# obj$(VERSION)/%.o:  src/%.c
# 	$(CC) -c -o obj$(VERSION)/$*.o $(C_FLAGS) $<
# 
