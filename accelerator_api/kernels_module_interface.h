/**
 * Kernels Module Runtime Interface
 *
 * This provides an interface through which to find and utilize GPU
 * compute kernels that are not visible to the caller's compilation block.
 * The interface supports an application dynamically loading a shared
 * library during runtime satisfying the interface declared here.
 * 
 * 
 * extern "C" designation is motivated to avoid C++ name mangling between the shared
 * library and the library consumer.
 * TODO: I understand everything marked extern "C" should be noexcept, therefore come up
 * a more reasonable error handling scheme. perhaps an (optional) C++ callback class ref/ptr
 */

#ifndef KERNELS_MODULE_RUNTIME_H
#define KERNELS_MODULE_RUNTIME_H

class KernelFn;


/**
 * Interface for GPU kernels found and launched across an opaque interface such as
 * runtime loaded shared library.
 */

extern "C"  {

	/**
	 * Access device kernel function point by character string name.
	 */
	const KernelFn * get_kernel_by_name(const char* kernel_name) noexcept;

	/**
	 * Issues a new work segment with one-dimensional specification to the device and queue.
	 * TODO: size_t sharedMem = 0, cudaStream_t stream = 0
	 */
	void enqueueKernelWork_1D( const KernelFn *, int numBlocks, int blockSize, void** args) noexcept;

}


#endif //KERNELS_MODULE_RUNTIME_H
