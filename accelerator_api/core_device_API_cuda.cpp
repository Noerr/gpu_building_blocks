/**
 * Prototype separated GPU kernels and the main driver program.
 * Ultimate goal is to run-time link GPU kernels from a shared library.
 *
 */

#include "compute_device_core_API.h"
#include "stream_implementation_cuda.h"
#include <cuda_runtime.h>
#include <nvToolsExtCuda.h> //for the nvtx profiler annotations api.

#include <sstream>
#include <stdexcept>
#include <vector>

//for the uuid print
#include <tuple>
#include <iomanip>


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

//TODO:  no actual state association with ComputeDevice class.  that class hasn't been designed to have instances yet.
void setComputeDevice(int device_numerator)
{
	int device_count;
	cudaError_t res = cudaGetDeviceCount ( &device_count );
	cudaError_t ret  = cudaSetDevice ( device_numerator % device_count ); throw_on_cuda_error(ret, __FILE__, __LINE__);
}

//source: https://stackoverflow.com/questions/68823023/set-cuda-device-by-uuid
std::string uuid_print(cudaUUID_t a){
  std::stringstream ss;
  std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
  std::string sep="";
  for (auto t : r){
    ss << sep;
    for (int i = std::get<0>(t); i < std::get<1>(t); i++)
      ss << std::hex << std::setfill('0') << std::setw(2) << (unsigned)(unsigned char)a.bytes[i];
	sep="-";
  }
  return ss.str();
}

std::string
ComputeDevice::getGPUDeviceInfoString()
{
	std::stringstream ss;
	int device_count, current_device;
	cudaError_t res = cudaGetDeviceCount ( &device_count );
	cudaError_t res2 = cudaGetDevice ( &current_device );
	//cudaDevAttrPciBusId: PCI bus identifier of the device
	cudaDeviceProp gpuProperties;
	cudaError_t res3 = cudaGetDeviceProperties ( &gpuProperties, current_device );

	ss << "Device " << current_device << " of " << device_count << " visible devices: id=" << uuid_print(gpuProperties.uuid) << " " << gpuProperties.name;
	return ss.str();
}

DeviceStream *
ComputeDevice::createStream()
{
	DeviceStream * p_newstream = new DeviceStream(*this);
	return p_newstream;
}
	
void
ComputeDevice::freeStream(DeviceStream * p_stream)
{
	delete p_stream;
}

void * 
ComputeDevice::malloc( std::size_t numBytes )
{
	void * allocation;
	throw_on_cuda_error( cudaMalloc ( &allocation, numBytes ), __FILE__, __LINE__);
	//TODO: why does this not specify a cuda device?  Consider adding cudaMemAdvise call for case of multi-GPU systems.++++
	//TODO: ??? cudaMemAttachGlobal );
	return allocation;
}


void
ComputeDevice::free( void * allocation_pointer )
{
	throw_on_cuda_error( cudaFree ( allocation_pointer ), __FILE__, __LINE__);
}


DeviceStream::DeviceStream(ComputeDevice& parentDevice)
{
	cudaStream_t newStream;
	unsigned int stream_flags = cudaStreamNonBlocking;
	cudaError_t ret =
	cudaStreamCreateWithFlags (&newStream, stream_flags); throw_on_cuda_error(ret, __FILE__, __LINE__);
	_pimpl = new Strm_Impl(newStream);
}

DeviceStream::~DeviceStream()
{
	cudaError_t ret =
	cudaStreamDestroy (_pimpl->get_cudaStream()); throw_on_cuda_error(ret, __FILE__, __LINE__);
	delete _pimpl;
}


void * 
DeviceStream::memcpy(void *__restrict__ dst, const void *__restrict__ src, std::size_t numBytes)
{

	cudaMemcpyKind kind = cudaMemcpyDefault;  //TODO:  more direction consideration. (look at HIP and SYCL)
	/* kind specifies the direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing. */
	//TODO:  likely change to cudaMemcpyAsync which takes a stream * argument.	
	
	cudaError_t ret1 =
	cudaMemcpyAsync( dst, src, numBytes, kind, _pimpl->get_cudaStream());  throw_on_cuda_error(ret1, __FILE__, __LINE__);
	return dst;
}


void
DeviceStream::sync()
{
	cudaError_t ret1 =
	cudaStreamSynchronize(_pimpl->get_cudaStream());  throw_on_cuda_error( ret1 , __FILE__, __LINE__);	
}


namespace Tracing {
	void traceRangePush(const char * traceRegionLabel)
	{
		nvtxRangePush(traceRegionLabel);
	}

	void traceRangePop()
	{
		nvtxRangePop();
	}
	
}
