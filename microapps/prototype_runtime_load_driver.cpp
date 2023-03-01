/**
 * Prototype separated GPU kernels and the main driver program.
 * Ultimate goal is to run-time link GPU kernels from a shared library.
 *
 */


//----------------------------------------------------------------------------------------
//   DRIVER

#include "compute_device_core_API.h"
#include "kernels_module_interface.h"
#include "distributed_variable.h"


#include <dlfcn.h> //for the runtime dlopen and related functionality
//#include <stdio.h>
#include <vector>
#include <sstream>
#include <iostream>



typedef size_t GO;  typedef short LO;




/*
 * arg[1] - number of elements per process (default 8)
  */
int main(int argc, char *argv[]) {

    if (argc > 5 || argc < 4) {
    std::cerr << "Exactly 3 arguments expected:   numElements  runtimeModuleName.so  kernelName" <<std::endl;
    return 1;
    }
    
    int numElemPerProcess = 8;
    if (argc > 1) {
        std::stringstream arg1(argv[1]);
        arg1 >> numElemPerProcess;
    }

    std::string runtime_module_filename;
    if (argc > 2) {
        std::stringstream arg2(argv[2]);
        arg2 >> runtime_module_filename;
    }

    std::string kernel_name;
    if (argc > 3) {
        std::stringstream arg3(argv[3]);
        arg3 >> kernel_name;
    }

    void * libhandle = dlopen(runtime_module_filename.c_str(), RTLD_NOW | RTLD_DEEPBIND ); //longterm preference: RTLD_LAZY);
    if (libhandle == nullptr) {
        std::cerr << "Error with dlopen " << runtime_module_filename.c_str() <<std::endl << dlerror() << std::endl;
        return 1;
    }

    
    typedef decltype(&get_kernel_by_name) get_kernel_by_name_fn_t;
    typedef decltype(&enqueueKernelWork_1D) enqueueKernelWork_fn_t;

    const char* dlerr_str;
    
    dlerr_str = dlerror();
    get_kernel_by_name_fn_t get_kernel_by_name_module1 = reinterpret_cast<get_kernel_by_name_fn_t>(dlsym(libhandle, "get_kernel_by_name") );
    dlerr_str = dlerror();
    if (dlerr_str != NULL) {
        std::cerr << "Error with dlsym: " <<  dlerr_str << std::endl;
        return 1;
    }

    dlerror();
    enqueueKernelWork_fn_t enqueueKernelWork_module1 = reinterpret_cast<enqueueKernelWork_fn_t>(dlsym(libhandle, "enqueueKernelWork_1D"  ) );
    dlerr_str = dlerror();
    if (dlerr_str != NULL) {
        std::cerr << "Error with dlsym: " <<  dlerr_str << std::endl;
        return 1;
    }


    int numranks=1; int myrank=0;
    
    
    std::vector<GO> myGlobalOrdinals(numElemPerProcess);
    for ( int i=0; i< myGlobalOrdinals.size(); i++ )
        myGlobalOrdinals[i] = myrank*numElemPerProcess*10 + i + 42; // just some funny numbering to reinforce no need that all GO's be sequential.
    
    
    
    DistributedVariable<double, LO, GO> myFaces( myGlobalOrdinals );
    DistributedVariable<double, LO, GO> yourFaces( myGlobalOrdinals );
    
    // Initialize the variable elements by GPU kernel
    int blockSize = 256;
    int numBlocks = (numElemPerProcess + blockSize - 1) / blockSize;

    void * my_faces_dev_ptr = myFaces.device_ptr();
	void * your_faces_dev_ptr = yourFaces.device_ptr();

    // kernel run #1 : initialize elements based on runtime argument
    void * args1[] = {&numElemPerProcess, &my_faces_dev_ptr, &myrank};
    const KernelFn * user_choice_kernel = get_kernel_by_name_module1( kernel_name.c_str() );
    enqueueKernelWork_module1( user_choice_kernel, numBlocks, blockSize, args1);
    
	// kernel run #2 : copy elements
    void * args2[] = {&numElemPerProcess, &my_faces_dev_ptr, &your_faces_dev_ptr};
    const KernelFn * copy_element_kernel = get_kernel_by_name_module1( "copy_element_kernel" );
    enqueueKernelWork_module1( copy_element_kernel, numBlocks, blockSize, args2);

    std::vector<double> host_result(numElemPerProcess);
    
    device_memcpy( &host_result[0], yourFaces.device_ptr(), numElemPerProcess*sizeof(double));
    
    device_sync();
    
    const char* sep = "";
    for (int p = 0; p < numranks; p++)
    {
        if (p==myrank) {
            std::cout << "P " << p << ": ";
            for (int e=0; e<numElemPerProcess; e++) {
                std::cout << sep << host_result[e];
                sep=", ";
            }
            std::cout << std::endl;

        }
    }


    int dlclose_results = dlclose(libhandle);

    return 0;
}

