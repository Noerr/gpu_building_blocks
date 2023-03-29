/**
 * implementation needed by the kernel module's in hip
 * 
 */

#ifndef STREAM_IMPL_HIP_H
#define STREAM_IMPL_HIP_H


#include <hip/hip_runtime.h>
#include "compute_device_core_API.h"


class DeviceStream::Strm_Impl
{
	hipStream_t _hip_stream;
	public:
		Strm_Impl(hipStream_t newStream)
		: _hip_stream(newStream)
		{
		}
		hipStream_t & get_hipStream()
		{
			return _hip_stream;
		}
};


#endif // STREAM_IMPL_HIP_H