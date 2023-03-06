/**
 * Prototype separated GPU kernels and the main driver program.
 * Ultimate goal is to run-time link GPU kernels from a shared library.
 *
 */

#include "compute_device_core_API.h"
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <vector>


void throw_on_cuda_error( cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    std::stringstream ss;
    ss << "CUDA Runtime Error " << code << " at " << file << "(" << line << ")";
    throw std::runtime_error(ss.str());
  }
}

namespace {
	ComputeDevice _one_and_only_CUDA_device;
	
}

ComputeDevice & getComputeDevice()
{
	return _one_and_only_CUDA_device;

}

void * 
ComputeDevice::malloc( std::size_t numBytes )
{
	void * allocation;
	throw_on_cuda_error( cudaMalloc ( &allocation, numBytes ), __FILE__, __LINE__);
	//TODO: why does this not specify a cuda device?  Consider adding cudaMemAdvise call for case of multi-GPU systems.++++
	return allocation;
}


void
ComputeDevice::free( void * allocation_pointer )
{
	throw_on_cuda_error( cudaFree ( allocation_pointer ), __FILE__, __LINE__);
}


void * 
ComputeDevice::DeviceStream::memcpy(void *restrict dst, const void *restrict src, std::size_t numBytes)
{

	cudaMemcpyKind kind = cudaMemcpyDefault;  //TODO:  more direction consideration. (look at HIP and SYCL)
	/* kind specifies the direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing. */
	//TODO:  likely change to cudaMemcpyAsync which takes a stream * argument.	
	
	cudaError_t ret1 =
	cudaMemcpy( dst, src, numBytes, kind);  throw_on_cuda_error(ret1, __FILE__, __LINE__);
	return dst;
}


void
ComputeDevice::DeviceStream::sync()
{
	cudaError_t ret1 =
	cudaDeviceSynchronize();  throw_on_cuda_error( ret1 , __FILE__, __LINE__);
	// TODO: replace cudaStreamSynchronize ( cudaStream_t stream )
	
}

