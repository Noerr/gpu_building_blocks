/**
 * CUDA kernel device functions part of prototyping separation of kernels to shared lib and runtime program
 *
 * after several tests, I found no working solution to separate CUDA kernel definitions in a shared library
 * with cudaLaunchKernel(...) in a separate compilation block (linking to that library).
 *    cudaErrorSymbolNotFound = 500
 *    This indicates that a named symbol was not found. Examples of symbols are global/constant variable names,
 *    driver function names, texture names, and surface names.
 * The CUDA runtime implementations utilizes kernel function pointers as a more advanced key into a structure of
 * properties for the kernel.  For this to work, it seems kernel launch needs to see the kernel definition.
 * There is likely a way to separate the to using the lower-level cu* device library.  At this time, I don't see
 * any reason to explore that approach.
 */
 
/*
nvcc --shared -o lib_kernel_parts.so cuda_kernel_parts.cu --compiler-options '-fPIC'
*/

#include <cuda_runtime.h>
#include <map>
#include <sstream>
#include <stdexcept>

#include <stdio.h> //debugging

#include "kernels_module_interface.h"


typedef size_t GO;  typedef short LO;

__global__
void initialize_element_kernel(LO lid, double *e , int myProcessID)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < lid; i += stride)
    e[i] = myProcessID*10000 + i;
}

__global__
void copy_element_kernel(LO lid, const double *e_src, double *e_dst)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < lid; i += stride)
      e_dst[i] = e_src[i] + 1000;
}


class KernelFn
{
public:
	
	const void * toCUDA_kernel_fn() const
	{
		return _mbr_generic_ptr;
	}

	KernelFn(const void* fn_ptr) : _mbr_generic_ptr(fn_ptr) {}

private:
	const void * _mbr_generic_ptr;
	
};


namespace {

void throw_on_cuda_error( cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    std::stringstream ss;
    ss << "CUDA Runtime Error " << code << " at " << file << "(" << line << ")";
    throw std::runtime_error(ss.str());
  }
}



template <typename CUDA_fn_type>
KernelFn makeGeneric( CUDA_fn_type cuda_kernel_fn)
{
	return KernelFn(static_cast<const void *>(cuda_kernel_fn));
}


class KernelStore
{
public:
	const KernelFn* get( const std::string& kernel_name )
	{
		return _kernel_map.at(kernel_name);
	}
	
	
	KernelStore()
	{
		load_more();
	}
	
	~KernelStore()
	{
		name_to_GPU_kernel_map_t::iterator itr = _kernel_map.begin();
		while (itr != _kernel_map.end()) {
			delete itr->second;
			itr = _kernel_map.erase(itr);
		}
	}
	
private:

	typedef std::map<std::string,  const KernelFn * >  name_to_GPU_kernel_map_t;
	
	
	void load_more()
	{
		//TODO: move (this repetitive) init to dynamic load of library ... 
		void * test1 = reinterpret_cast<void*>(initialize_element_kernel);
		_kernel_map["initialize_element_kernel"] = new KernelFn(makeGeneric( test1 ));
		printf("initialize_element_kernel:  %p ,  %p\n",  initialize_element_kernel,  _kernel_map.at("initialize_element_kernel")->toCUDA_kernel_fn() );
		
		void * test2 = reinterpret_cast<void*>(copy_element_kernel);
		_kernel_map["copy_element_kernel"]       = new KernelFn( makeGeneric(test2));
	}
	
	
	name_to_GPU_kernel_map_t _kernel_map;
};

KernelStore _kernel_store;
} // unnamed namespace


extern "C"  {

const KernelFn *
get_kernel_by_name(const char* kernel_name) noexcept
{

	return _kernel_store.get(kernel_name);
}


void
enqueueKernelWork_1D( const KernelFn * fn, int numBlocks, int blockSize, void** args) noexcept
{
	const void * cuda_kernel_fn = fn->toCUDA_kernel_fn(); 
	
	dim3 gridDim3(numBlocks);
    dim3 blockSize3(blockSize);
	
	printf("launching fn by ptr %p\n", cuda_kernel_fn);  fflush(stdout);
	cudaError_t ret1 = 
    cudaLaunchKernel( cuda_kernel_fn, gridDim3, blockSize3, args, 0 /*shared*/, 0 /*stream*/ );
    throw_on_cuda_error( ret1 , __FILE__, __LINE__);

}


}
