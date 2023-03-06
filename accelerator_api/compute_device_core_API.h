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


ComputeDevice & getComputeDevice();


#endif //COMPUTE_DEVICE_CORE_API_H
