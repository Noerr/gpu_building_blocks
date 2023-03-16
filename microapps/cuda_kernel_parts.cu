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
#include "stream_implementation_cuda.h"

// bwah. ideally the kernel definitions only appear once, but i wasted two much time trying to sort out a preprocessor way to achieve this aim.
#if defined(KERNEL_LINK_METHOD_RTC)
#include <nvrtc.h>
#include <cuda.h>
namespace {
const char* _kernels_string = R"rawstring(

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

)rawstring";
}
#else

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
#endif

class KernelFn
{
public:

#if defined(KERNEL_LINK_METHOD_RTC)

	CUfunction to_CU_fn() const
	{
		return _mbr_cufn;
	}

	KernelFn( CUfunction cufn )
	: _mbr_cufn(cufn)
	{
	}
	private:
	CUfunction _mbr_cufn;
#else
	const void * toCUDA_kernel_fn() const
	{
		return _mbr_generic_ptr;
	}


	KernelFn(const void* fn_ptr) : _mbr_generic_ptr(fn_ptr) {}

	private:
	const void * _mbr_generic_ptr;
#endif

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
	
	//TODO: move (this repetitive) init to dynamic load of library ...
	void load_more();

		
	name_to_GPU_kernel_map_t _kernel_map;
};

KernelStore _kernel_store;
} // unnamed namespace


void
KernelStore::load_more()
{
#if defined(KERNEL_LINK_METHOD_RTC)
	//build the kernels from _kernels_string
	// Create an instance of nvrtcProgram
	nvrtcProgram prog;
	nvrtcCreateProgram(&prog,         // prog
						_kernels_string,  // src_string
						__FILE__,    // filename proxy, can be NULL
						0,             // numHeaders
						NULL,          // headers
						NULL);        // includeNames
	// Compile the program with fmad disabled.
	// Note: Can specify GPU target architecture explicitly with '-arch' flag.
	const char *opts[] = {"--fmad=false"};
	nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
													1,     // numOptions
													opts); // options
	// Obtain compilation log from the program.
	size_t logSize;
	nvrtcGetProgramLogSize(prog, &logSize);
	if (logSize>1) {
		char *log = new char[logSize];
		nvrtcGetProgramLog(prog, log);
		printf("NVRTC Compile Log: \n%s\n", log);
		delete[] log;
	}
	throw_on_cuda_error( (compileResult == NVRTC_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue, __FILE__, __LINE__);
	
	// Obtain PTX from the program.
	size_t ptxSize;
	nvrtcGetPTXSize(prog, &ptxSize);
	char *ptx = new char[ptxSize];
	nvrtcGetPTX(prog, ptx);
	// Destroy the program.
	nvrtcDestroyProgram(&prog);
	// Load the generated PTX and get a handle to the SAXPY kernel.
	//CUdevice cuDevice;
	//CUcontext context;
	CUmodule module;
	CUfunction cu_fn_initialize_element_kernel;
	CUfunction cu_fn_copy_element_kernel;
	//CUDA_SAFE_CALL(cuInit(0));
	//CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
	//CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
	cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	cuModuleGetFunction(&cu_fn_initialize_element_kernel, module, "initialize_element_kernel");
	cuModuleGetFunction(&cu_fn_copy_element_kernel, module, "copy_element_kernel");
	delete[] ptx;

	// questionable deviation from cu to cuda API
	//void * test1 = reinterpret_cast<void*>(cu_fn_initialize_element_kernel);
	//void * test2 = reinterpret_cast<void*>(cu_fn_copy_element_kernel);
	//TODO: move (this repetitive) init to dynamic load of library ... 
	_kernel_map["initialize_element_kernel"] = new KernelFn( cu_fn_initialize_element_kernel );
	_kernel_map["copy_element_kernel"]       = new KernelFn( cu_fn_copy_element_kernel );
	
#else 
		void * fn1 = reinterpret_cast<void*>(initialize_element_kernel);
		void * fn2 = reinterpret_cast<void*>(copy_element_kernel);

		_kernel_map["initialize_element_kernel"] = new KernelFn( fn1 );
		_kernel_map["copy_element_kernel"]       = new KernelFn( fn2 );
#endif

}

extern "C"  {

const KernelFn *
get_kernel_by_name(const char* kernel_name) noexcept
{
	return _kernel_store.get(kernel_name);
}


void
enqueueKernelWork_1D( DeviceStream* pStream, const KernelFn * fn, int numBlocks, int blockSize, void** args) noexcept
{ 
	
	dim3 gridDim3(numBlocks);
    dim3 blockSize3(blockSize);

	cudaStream_t cudaStream = to_impl_type<cudaStream_t>( pStream );

#if defined(KERNEL_LINK_METHOD_RTC)
	CUfunction cu_kernel_fn = fn->to_CU_fn();
	cuLaunchKernel(cu_kernel_fn,
                   numBlocks, 1, 1,    // grid dim
                   blockSize, 1, 1,   // block dim
                   0, cudaStream /* casting cudaStream_t to CUstream */,             // shared mem and stream
                   args, 0);  
#else
	const void * cuda_kernel_fn = fn->toCUDA_kernel_fn();
	//printf("launching fn by ptr %p\n", cuda_kernel_fn);  fflush(stdout);
	cudaError_t ret1 = 
    cudaLaunchKernel( cuda_kernel_fn, gridDim3, blockSize3, args, 0 /*shared*/, cudaStream ); throw_on_cuda_error( ret1 , __FILE__, __LINE__);
#endif

}


}
