/**
 * implementation needed by the kernel module's in cuda
 * 
 */

#ifndef STREAM_IMPL_CUDA_H
#define STREAM_IMPL_CUDA_H


#include <cuda_runtime.h>
#include "compute_device_core_API.h"


class DeviceStream::Strm_Impl
{
	cudaStream_t _cuda_stream;
	public:
		Strm_Impl(cudaStream_t newStream)
		: _cuda_stream(newStream)
		{
		}
		cudaStream_t & get_cudaStream()
		{
			return _cuda_stream;
		}
};


template <>
cudaStream_t to_impl_type(DeviceStream* p_stream)
{
	return p_stream->_pimpl->get_cudaStream();
}

#endif // STREAM_IMPL_CUDA_H