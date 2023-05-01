#include "hip/hip_runtime.h"
/**
 * HIP kernel device functions part of prototyping separation of kernels to shared lib and runtime program
 *
 * after several tests, I found no working solution to separate HIP kernel definitions in a shared library
 * with hipLaunchKernel(...) in a separate compilation block (linking to that library).
 *    hipErrorNotFound = 500
 *    This indicates that a named symbol was not found. Examples of symbols are global/constant variable names,
 *    driver function names, texture names, and surface names.
 * The HIP runtime implementations utilizes kernel function pointers as a more advanced key into a structure of
 * properties for the kernel.  For this to work, it seems kernel launch needs to see the kernel definition.
 * There is likely a way to separate the to using the lower-level cu* device library.  At this time, I don't see
 * any reason to explore that approach.
 */
 
/*
nvcc --shared -o lib_kernel_parts.so kernel_parts.hip --compiler-options '-fPIC'
*/

#include <hip/hip_runtime.h>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

#include <stdio.h> //debugging

#include "kernels_module_interface.h"
#include "stream_implementation_hip.h"

#if defined(KERNEL_LINK_METHOD_RTC)
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#endif


////////////////////////////////////////////////////////////////////////////
// Below we use macro KERNEL_CODE_HANDLER() enclosing the GPU kernel code
// to either do nothing or convert the whole block to a string literal
// at compile time for the runtime compilation (RTC) method.
// Note that for RTC, newlines are removed by the STRINGIZE macro,
// so all comments must be in /* C-style */.
// This complexity is simply so that this same source file can support muliple
// kernel compilation methods while writing the kernel code just once.
////////////////////////////////////////////////////////////////////////////
#if defined(KERNEL_LINK_METHOD_RTC)
#define STRINGIZE(A) #A
#define KERNEL_CODE_HANDLER(A)  STRINGIZE(A)
namespace {
const char* _kernels_string = 
#else
#define KERNEL_CODE_HANDLER(A)  A
#endif

KERNEL_CODE_HANDLER(
typedef size_t GO;  typedef short LO;

__global__
void initialize_element_kernel(int numLocalItems, double *e , const GO * globalOrdinals, int myProcessID)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  // just a funny marker to each item to allow for inspecting uniqueness and ownership
  for (int i = index; i < numLocalItems; i += stride)
    e[i] = myProcessID*0.0001 + globalOrdinals[i];
}

__global__
void copy_element_kernel(int numLocalItems, const double *e_src, double *e_dst)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < numLocalItems; i += stride)
      e_dst[i] = e_src[i] + 1000;
}

__global__
void average_elements_kernel(int numLocalItems, const double *e_src, double *e_dst)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < numLocalItems; i += stride)
      e_dst[i] = 0.5*(e_dst[i] + e_src[i]);
}


__global__
void sum_reduce_kernel(int numLocalItems, const double *e, double * localSum )
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // int stride = blockDim.x * gridDim.x;
  // TODO:  come back and write this proper for GPU. (void) stride;
  double mySum = 0.;
  if (index==0) {
   for (int i = index; i < numLocalItems; i ++ )
      mySum += e[i];
   localSum[0] = mySum;
  }
}

) //KERNEL_CODE_HANDLER()
#if defined(KERNEL_LINK_METHOD_RTC)
; }
#endif
////////////////////////////////////////////////////////////////////////////



class KernelFn
{
public:

#if defined(KERNEL_LINK_METHOD_RTC)

	hipFunction_t to_HIP_fn() const
	{
		return _mbr_hipfn;
	}

	KernelFn( hipFunction_t hipfn )
	: _mbr_hipfn(hipfn)
	{
	}
	private:
	hipFunction_t _mbr_hipfn;
#else
	const void * toHIP_kernel_fn_ptr() const
	{
		return _mbr_generic_ptr;
	}


	KernelFn(const void* fn_ptr) : _mbr_generic_ptr(fn_ptr) {}

	private:
	const void * _mbr_generic_ptr;
#endif

};


namespace {

void throw_on_hip_error( hipError_t code, const char *file, int line)
{
  if(code != hipSuccess)
  {
    std::stringstream ss;
    ss << "HIP Runtime Error " << code << " at " << file << "(" << line << "):" << hipGetErrorString(code);
    throw std::runtime_error(ss.str());
  }
}

#if defined(KERNEL_LINK_METHOD_RTC)
void throw_on_hiprtc_error( hiprtcResult result, const char *file, int line)
{
  if(result != HIPRTC_SUCCESS)
  {
    std::stringstream ss;
    ss << "HIP RTC Runtime Error " << result << " at " << file << "(" << line << "):" << hiprtcGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#endif


class KernelStore
{
public:
	const KernelFn* get( const std::string& kernel_name )
	{
		if (!_lateLoadComplete) {
			// changed to doing "late load" instead of the constructor after porting to HIP.
			// Some of the HIP RTC functions complained about HIP not initialized yet when load_more was running before main().
			load_more();
			_lateLoadComplete = true;
		}
		return _kernel_map.at(kernel_name);
	}
	
	
	KernelStore()
	: _lateLoadComplete(false)
	{
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
	bool _lateLoadComplete;
};

KernelStore _kernel_store;
} // unnamed namespace


void
KernelStore::load_more()
{
#if defined(KERNEL_LINK_METHOD_RTC)
	//build the kernels from _kernels_string
	// Create an instance of hiprtcProgram
	hiprtcProgram prog;
	hiprtcResult createResult = hiprtcCreateProgram(&prog,         // prog
						_kernels_string,  // src_string
						__FILE__,    // filename proxy, can be NULL
						0,             // numHeaders
						NULL,          // headers
						NULL);        // includeNames
	throw_on_hiprtc_error( createResult, __FILE__, __LINE__);
	

	// demangling of kernel names needed.   HIP provides utility functions to do this
	std::vector<std::string> kernel_name_vec;
	kernel_name_vec.push_back("initialize_element_kernel");
	kernel_name_vec.push_back("copy_element_kernel");
	kernel_name_vec.push_back("average_elements_kernel");
	kernel_name_vec.push_back("sum_reduce_kernel");
	for (auto&& x : kernel_name_vec) throw_on_hiprtc_error( hiprtcAddNameExpression(prog, x.c_str()), __FILE__, __LINE__);
	
	// Compile the program with fmad disabled.
	// Note: Can specify GPU target architecture explicitly with '-arch' flag.
	const char *opts[] = {}; //cuda only:{"--fmad=false",};
	hiprtcResult compileResult = hiprtcCompileProgram(prog,  // prog
													sizeof(opts)/sizeof(char*),     // numOptions
													opts); // options
	// Obtain compilation log from the program.
	size_t logSize;
	throw_on_hiprtc_error( hiprtcGetProgramLogSize(prog, &logSize), __FILE__, __LINE__ );
	if (logSize>1) {
		char *log = new char[logSize];
		hiprtcGetProgramLog(prog, log);
		printf("RTC Compile Log: \n%s\n", log);
		delete[] log;
	}
	throw_on_hiprtc_error( compileResult, __FILE__, __LINE__);
	
	// Obtain device code from the program.
	size_t deviceCodeSize;
	throw_on_hiprtc_error( hiprtcGetCodeSize(prog, &deviceCodeSize), __FILE__, __LINE__);
	char *deviceCode = new char[deviceCodeSize];
	throw_on_hiprtc_error( hiprtcGetCode(prog, deviceCode), __FILE__, __LINE__);
	//printf("kernel code size is %zu Bytes.\n------------------\n%s", deviceCodeSize, deviceCode);  fflush(stdout);
	

	// get the lowered (mangled) names
	const char * mangled_initialize_element_kernel_str;
	const char * mangled_copy_element_kernel_str;
	const char * mangled_average_elements_kernel_str;
	const char * mangled_sum_reduce_kernel_str;
	throw_on_hiprtc_error( hiprtcGetLoweredName(prog, kernel_name_vec[0].c_str(), &mangled_initialize_element_kernel_str), __FILE__, __LINE__);
	throw_on_hiprtc_error( hiprtcGetLoweredName(prog, kernel_name_vec[1].c_str(), &mangled_copy_element_kernel_str), __FILE__, __LINE__);
	throw_on_hiprtc_error( hiprtcGetLoweredName(prog, kernel_name_vec[2].c_str(), &mangled_average_elements_kernel_str), __FILE__, __LINE__);
	throw_on_hiprtc_error( hiprtcGetLoweredName(prog, kernel_name_vec[3].c_str(), &mangled_sum_reduce_kernel_str), __FILE__, __LINE__);
	
	// printf("mangled names are: \n%s\n%s\n", mangled_initialize_element_kernel_str, mangled_copy_element_kernel_str);  fflush(stdout);
	
	// Load the generated deviceCode
	hipModule_t module;
	hipFunction_t hip_fn_initialize_element_kernel;
	hipFunction_t hip_fn_copy_element_kernel;
	hipFunction_t hip_fn_average_elements_kernel;
	hipFunction_t hip_fn_sum_reduce_kernel;
	
	throw_on_hip_error(hipModuleLoadDataEx(&module, deviceCode, 0, 0, 0), __FILE__, __LINE__);
	throw_on_hip_error(hipModuleGetFunction(&hip_fn_initialize_element_kernel, module, mangled_initialize_element_kernel_str), __FILE__, __LINE__);
	throw_on_hip_error(hipModuleGetFunction(&hip_fn_copy_element_kernel, module, mangled_copy_element_kernel_str), __FILE__, __LINE__);
	throw_on_hip_error(hipModuleGetFunction(&hip_fn_average_elements_kernel, module, mangled_average_elements_kernel_str), __FILE__, __LINE__);
	throw_on_hip_error(hipModuleGetFunction(&hip_fn_sum_reduce_kernel, module, mangled_sum_reduce_kernel_str), __FILE__, __LINE__);

	// Destroy the program  (AFTER utilizing mangled name strings)
	hiprtcDestroyProgram(&prog);
	delete[] deviceCode;

	_kernel_map["initialize_element_kernel"] = new KernelFn( hip_fn_initialize_element_kernel );
	_kernel_map["copy_element_kernel"]       = new KernelFn( hip_fn_copy_element_kernel );
	_kernel_map["average_elements_kernel"]   = new KernelFn( hip_fn_average_elements_kernel );
	_kernel_map["sum_reduce_kernel"]         = new KernelFn( hip_fn_sum_reduce_kernel );
	
#else 
		void * fn1 = reinterpret_cast<void*>(initialize_element_kernel);
		void * fn2 = reinterpret_cast<void*>(copy_element_kernel);
		void * fn3 = reinterpret_cast<void*>(average_elements_kernel);
		void * fn4 = reinterpret_cast<void*>(sum_reduce_kernel);

		_kernel_map["initialize_element_kernel"] = new KernelFn( fn1 );
		_kernel_map["copy_element_kernel"]       = new KernelFn( fn2 );
		_kernel_map["average_elements_kernel"]   = new KernelFn( fn3 );
		_kernel_map["sum_reduce_kernel"]         = new KernelFn( fn4 );
		
#endif

}


template <>
hipStream_t to_impl_type(DeviceStream* p_stream)
{
	return p_stream->_pimpl->get_hipStream();
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

	hipStream_t hipStream = to_impl_type<hipStream_t>( pStream );

#if defined(KERNEL_LINK_METHOD_RTC)
	hipFunction_t hip_kernel_fn = fn->to_HIP_fn();

#if defined(__HIP_PLATFORM_NVIDIA__)
	throw_on_hip_error(hipModuleLaunchKernel(hip_kernel_fn,
                   numBlocks, 1, 1,    // grid dim
                   blockSize, 1, 1,   // block dim
                   0, hipStream /* casting hipStream_t to hipStream_t */,             // shared mem and stream
                   args, 0), __FILE__, __LINE__);
#elif defined(__HIP_PLATFORM_AMD__)
	//Discovered: "Warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please refer to hip_porting_driver_api.md for sample usage."
	size_t args_size = 4*sizeof(void*); // <- this is problematic hard-coding number of arguments.
	// TODO: I did not write infrastructure to track number of kernel arguments.  This one case of ROCm + RTC is not currently capable to understand number of arguments inherently 
	void *launch_config[] = {
		HIP_LAUNCH_PARAM_BUFFER_POINTER, args, //&argBuffer[0],
		HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
		HIP_LAUNCH_PARAM_END
    };
	throw_on_hip_error(hipModuleLaunchKernel(hip_kernel_fn,
                   numBlocks, 1, 1,    // grid dim
                   blockSize, 1, 1,   // block dim
                   0, hipStream /* casting hipStream_t to hipStream_t */,             // shared mem and stream
                   nullptr, (void**)&launch_config), __FILE__, __LINE__);
#else
#error "HIP Platform not set"
#endif

#else
	const void * p_HIP_kernel_fn = fn->toHIP_kernel_fn_ptr();
	//printf("launching fn by ptr %p\n", hip_kernel_fn);  fflush(stdout);
	hipError_t ret1 = 
    hipLaunchKernel( p_HIP_kernel_fn, gridDim3, blockSize3, args, 0 /*shared*/, hipStream ); throw_on_hip_error( ret1 , __FILE__, __LINE__);
#endif

}


}
