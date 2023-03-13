/**
 * Compute Device Core API
 *
 * This header defines an interface for core GPU API functions without
 * exposing the the specific vendor implementation (eg. CUDA, HIP, SYCL)
 * to the application using this interface.
 *
 * The motivation for this interface abstraction is to encourage more portable
 * implementation of the consuming application.
 * 
 */
 
#ifndef COMPUTE_DEVICE_CORE_API_H
#define COMPUTE_DEVICE_CORE_API_H

#include <cstddef>
#include <string>

class DeviceStream;

/**
 * Abstraction of an asynchronous kernel chomping compute device (eg. CPU, GPU, accelerator)
 */
class ComputeDevice
{
	public:
	
	/**
	 * Allocate numBytes memory on the device.
	 */
	void * malloc( std::size_t numBytes );

	/**
	 * Free previous allocation by malloc member function.
	 */
	void free( void * allocation_pointer );
			
	
	DeviceStream * createStream();
	
	void freeStream(DeviceStream * p_stream);
	
	/**
	 * @brief Identifying information about the GPU.
	 * TODO:  static because I have not implemented and do not necessarily want to support multiple GPU per PE.
	 */
	static std::string getGPUDeviceInfoString();
	
};

template<typename TO_IMPL_TYPE>
TO_IMPL_TYPE to_impl_type(DeviceStream* p_stm);

/**
 * Abstraction of a asynchronous queue of work
 */
class DeviceStream
{
	DeviceStream(ComputeDevice& parentStream);
	friend class ComputeDevice;

	/**
	 * not satisfied with the object relationships in this one aspect where the 
	 * kernel module also needs to know about the underlying stream implementation
	 */
	template<typename TO_IMPL_TYPE>
	friend TO_IMPL_TYPE to_impl_type(DeviceStream* p_stm);

	class Strm_Impl;
	Strm_Impl* _pimpl;

	public:

		~DeviceStream();

		/**
		 * Synchronize (block) on the specified queue
		 */
		void sync();
		
		/**
		 * Copy memory to/from device.  Copies numBytes from memory area src to memory area dst.
		 * If dst and src overlap, behavior is undefined.
		 * 
		 * RETURN VALUES
		 *   The memcpy() function returns the original value of dst.
		 */
		void * memcpy(void * __restrict__ dst, const void * __restrict__ src, std::size_t numBytes);


};

/**
 * @brief Get the current or default Compute Device object
 * 
 * @return ComputeDevice& 
 */
ComputeDevice & getComputeDevice();

/**
 * @brief Set the current Compute Device object by rotating device index function
 * 
 * The funciton is device_numerator % device_count
 * @param device_numerator 
 */
void setComputeDevice(int device_numerator);

namespace Tracing {
	void traceRangePush(const char * traceRegionLabel);
	void traceRangePop();
}

#endif //COMPUTE_DEVICE_CORE_API_H
