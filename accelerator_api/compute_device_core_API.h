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
 
#include <cstddef>

#ifndef COMPUTE_DEVICE_CORE_API_H
#define COMPUTE_DEVICE_CORE_API_H




/**
 * Access device kernel function point by character string name.
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
	
		

	class DeviceStream
	{
		DeviceStream();
		~DeviceStream();
		
		public:
			/**
			 * Synchronize (block) on the specified queue
			 */
			sync();
			
			/**
			 * Copy memory to/from device.  Copies numBytes from memory area src to memory area dst.
			 * If dst and src overlap, behavior is undefined.
			 * 
			 * RETURN VALUES
			 *   The memcpy() function returns the original value of dst.
			 */
			void * memcpy(void *restrict dst, const void *restrict src, std::size_t numBytes);

	
	
	};
	
	DeviceStream * createStream();
	
	void freeStream(DeviceStream * p_stream);
	
	
};


ComputeDevice & getComputeDevice();


#endif //COMPUTE_DEVICE_CORE_API_H
