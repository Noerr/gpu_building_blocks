/**
 * Prototype separated GPU kernels and the main driver program.
 * Ultimate goal is to run-time link GPU kernels from a shared library.
 *
 */

#include "compute_device_core_API.h"
#include "stream_implementation_hip.h"
#include <hip/hip_runtime.h>

#if defined(__CUDA_API_VER_MAJOR__)  // substitute for expected HIP_PLATFORM_NVDIA
#define __HIP_PLATFORM_NVIDIA__ backup
#include <nvToolsExtCuda.h> //for the nvtx profiler annotations api.

#elif defined(__HIP_PLATFORM_AMD__)
#include <roctracer/roctx.h>

#else
#error "HIP Platform not set."
#endif

#include <sstream>
#include <stdexcept>
#include <vector>

//for the uuid print
#include <tuple>
#include <iomanip>


void throw_on_hip_error( hipError_t code, const char *file, int line)
{
  if(code != hipSuccess)
  {
    std::stringstream ss;
    ss << "HIP Runtime Error " << code << " at " << file << "(" << line << "):" << hipGetErrorString(code);
    throw std::runtime_error(ss.str());
  }
}

namespace {
	ComputeDevice _one_and_only_HIP_device;
	
}



ComputeDevice & getComputeDevice()
{
	return _one_and_only_HIP_device;

}

//TODO:  no actual state association with ComputeDevice class.  that class hasn't been designed to have instances yet.
void setComputeDevice(int device_numerator)
{
	int device_count;
	hipError_t res = hipGetDeviceCount ( &device_count );
	hipError_t ret  = hipSetDevice ( device_numerator % device_count ); throw_on_hip_error(ret, __FILE__, __LINE__);
}

//source: https://stackoverflow.com/questions/68823023/set-cuda-device-by-uuid
std::string uuid_print(hipUUID a){
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
	hipError_t res = hipGetDeviceCount ( &device_count );
	hipError_t res2 = hipGetDevice ( &current_device );
	//hipDeviceAttributePciBusId: PCI bus identifier of the device
	hipDeviceProp_t gpuProperties;
	hipError_t res3 = hipGetDeviceProperties ( &gpuProperties, current_device );
	hipUUID uuid;
	hipError_t res4 = hipDeviceGetUuid(	&uuid, current_device );	
	ss << "Device " << current_device << " of " << device_count << " visible devices: id=" << uuid_print(uuid) << " " << gpuProperties.name;
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
	throw_on_hip_error( hipMalloc ( &allocation, numBytes ), __FILE__, __LINE__);
	//TODO: why does this not specify a hip device?  Consider adding hipMemAdvise call for case of multi-GPU systems.++++
	//TODO: ??? hipMemAttachGlobal );
	return allocation;
}


void
ComputeDevice::free( void * allocation_pointer )
{
	throw_on_hip_error( hipFree ( allocation_pointer ), __FILE__, __LINE__);
}


DeviceStream::DeviceStream(ComputeDevice& parentDevice)
{
	hipStream_t newStream;
	unsigned int stream_flags = hipStreamNonBlocking;
	hipError_t ret =
	hipStreamCreateWithFlags (&newStream, stream_flags); throw_on_hip_error(ret, __FILE__, __LINE__);
	_pimpl = new Strm_Impl(newStream);
}

DeviceStream::~DeviceStream()
{
	hipError_t ret =
	hipStreamDestroy (_pimpl->get_hipStream()); throw_on_hip_error(ret, __FILE__, __LINE__);
	delete _pimpl;
}


void * 
DeviceStream::memcpy(void *__restrict__ dst, const void *__restrict__ src, std::size_t numBytes)
{

	hipMemcpyKind kind = hipMemcpyDefault;  //TODO:  more direction consideration. (look at HIP and SYCL)
	/* kind specifies the direction of the copy, and must be one of hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice, or hipMemcpyDefault. Passing hipMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, hipMemcpyDefault is only allowed on systems that support unified virtual addressing. */
	//TODO:  likely change to hipMemcpyAsync which takes a stream * argument.	
	
	hipError_t ret1 =
	hipMemcpyAsync( dst, src, numBytes, kind, _pimpl->get_hipStream());  throw_on_hip_error(ret1, __FILE__, __LINE__);
	return dst;
}


void
DeviceStream::sync()
{
	hipError_t ret1 =
	hipStreamSynchronize(_pimpl->get_hipStream());  throw_on_hip_error( ret1 , __FILE__, __LINE__);	
}


namespace Tracing {
	void traceRangePush(const char * traceRegionLabel)
	{
#if defined(__HIP_PLATFORM_NVIDIA__)
		nvtxRangePush(traceRegionLabel);
#elif defined(__HIP_PLATFORM_AMD__)
		roctxRangePush(traceRegionLabel);
#else
#error "HIP Platform not set."
#endif 
	}

	void traceRangePop()
	{
#if defined(__HIP_PLATFORM_NVIDIA__)
		nvtxRangePop();
#elif defined(__HIP_PLATFORM_AMD__)
		roctxRangePop();
#else
#error "HIP Platform not set."
#endif 
	}
	
}
