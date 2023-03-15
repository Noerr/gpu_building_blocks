/**
 * Prototype separated GPU kernels and the main driver program
 * combined with MPI use of GPU device pointers.
 *
 */


//----------------------------------------------------------------------------------------
//   DRIVER

#include "compute_device_core_API.h"
#include "kernels_module_interface.h"
#include "distributed_variable.h"


#ifdef KERNEL_LINK_METHOD_RUNTIME_MODULE
#include <dlfcn.h> //for the runtime dlopen and related functionality
#endif

#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <unistd.h> //for sleep() and getpid()
#include <assert.h>  // Define NDEBUG to disable

#include <mpi.h>


typedef size_t GO;
#ifndef LOCAL_ORDINAL_TYPE
#define LOCAL_ORDINAL_TYPE int
#endif
typedef LOCAL_ORDINAL_TYPE LO;


/**
 * pause if DEBUG_PAUSE_FOR_MPI_ID is set
 */
void debugger_attach_opportunity();

/**
 * Comma Separated List string
 */
template <typename T>
std::string makeCSLstr( std::vector<T> v );



///MOOOVE
/**
 * Returns prime factors sorted lowest to higest
 * Returns [1] for 1, otherwise 1 is omitted
 */
std::vector<int> primeFactors(unsigned int n);

/**
 * Returns a vector of length 3 with the product
 * of each element matching the product of all elements of factors
 */
std::vector<int> cubish(const std::vector<int> &factors);

/**
 * Calculate multi-index for flatIndex in a box  (C-style row-major ordering)
 */
std::vector<int> inverseRank( int flatIndex, const std::vector<int> &boxDim );

/**
 * Calculate flat-index for multi-index in a box  (C-style row-major ordering)
 */
int flatIndex( const std::vector<int> &multiIndex, const std::vector<int> &boxDim );

	/**
	 * Calculate local ordinal for a face normal to facedim.
	 *  Ordering is all faces normal to dim0, then dim1, then dim2.  C-style row-major ordering in those groups
	 *  "local" because upper extent faces on the box are not counted
	 */
template <typename LocalIndex>
LocalIndex localFaceOrdinal( int facedim, const std::vector<int> &multiIndex, const std::vector<int> &boxDim )
{
	// does not count upper box faces in each dim.
	int dim0Stride = boxDim[1]*boxDim[2];
	int dim1Stride = boxDim[2];
	int numFacesPerDim = boxDim[0]*dim0Stride;
	LocalIndex lo;
	if (facedim==0) {
		lo = multiIndex[0]*dim0Stride + multiIndex[1]*dim1Stride + multiIndex[2];
	} else if (facedim==1) {
		lo =  numFacesPerDim
		      + multiIndex[0]*dim0Stride + multiIndex[1]*dim1Stride + multiIndex[2];
	} else if (facedim==2) {
		lo =  2*numFacesPerDim
		      + multiIndex[0]*dim0Stride + multiIndex[1]*dim1Stride + multiIndex[2];
	}
	assert(lo >=0 && lo < 3*boxDim[0]*boxDim[1]*boxDim[2]);  // fails here have always been a max range issue for LocalIndex type.
	return lo;
}

// don't laugh. brain exercise. max range of interest ~1M.
std::vector<int> primeFactors(unsigned int n) {
	std::vector<int> factors;
	const int initial_n = n;
	if (n==1)
		return {1};
	int test=2;
	while (n > 1) {
		assert(test<=initial_n); // ("Prime Factorization implemenation FAIL!");
		if (n%test==0) {
			factors.push_back(test);
			n = n/test;
		} else
			test++;
	}
	
	return factors;
}

std::vector<int> cubish(const std::vector<int> &factors)
{
	std::vector<int> threeFactors = {1,1,1};
	for (int i=factors.size()-1; i>=0; i--)
		threeFactors[i%3] *= factors[i];
	return threeFactors;
}

std::vector<int> inverseRank( int flatIndex, const std::vector<int> &boxDim )
{
	std::vector<int> coord, stride(boxDim.size(), 1);
	for (int d=boxDim.size()-2; d>=0; d--)
		stride[d] = stride[d+1] * boxDim[d+1];
		
	for (int d=0; d<boxDim.size(); d++)
	{
		coord.push_back( flatIndex / stride[d] );
		flatIndex = flatIndex % stride[d];
	}
	return coord;
}

int flatIndex( const std::vector<int> &multiIndex, const std::vector<int> &boxDim )
{
	int index = 0;
	std::vector<int> stride(boxDim.size(), 1);
	for (int d=boxDim.size()-2; d>=0; d--)
		stride[d] = stride[d+1] * boxDim[d+1];
		
	for (int d=0; d<boxDim.size(); d++)
	{
		index += multiIndex[d] * stride[d];
	}
	return index;
}


template <typename LocalIndex, typename GlobalOrdinal_t, typename OffsetMap_t>
std::tuple< std::vector<GO>, OffsetMap_t, OffsetMap_t > 
createGlobalOrdinalDecomposition_3DstructuredFaces(int numElemPerProcess, int numProcesses, int myrank );

template <typename LocalIndex, typename GlobalOrdinal_t, typename OffsetMap_t>
std::tuple< std::vector<GO>, OffsetMap_t, OffsetMap_t > 
createGlobalOrdinalDecomposition_3DstructuredFaces(int numElemPerProcess, int numProcesses, int myrank )
{


	// Need to create a somewhat interesting arangement of 1:1 shared face relationships
	// implementing a 3D structured box
	std::vector<int> numRanksPrimeFactors =  primeFactors(numProcesses);
	std::vector<int> pdims3 = cubish( numRanksPrimeFactors );
	std::vector<int> numElementPrimeFactors =  primeFactors(numElemPerProcess);
	std::vector<int> edims3 = cubish( numElementPrimeFactors );
	std::vector<int> myPposition = inverseRank(myrank, pdims3);
	std::vector<int> yourPposition, yourEposition;
	
	if (myrank==0) {
		std::cout << "Process Grid Dims: " << makeCSLstr(pdims3) << std::endl;
		std::cout << "Cube Element Grid Dims (per process): " << makeCSLstr(edims3) << " [though something more like cube faces are exchanged]."<< std::endl;
	}
	
	// Working with FACES, not elements so about three unique faces per element, plus the global 3D box has one extra face per dim.
	size_t numFacesPerProcess = (edims3[0]+1)*(edims3[1]  )*(edims3[2]  )
	                        +(edims3[0]  )*(edims3[1]+1)*(edims3[2]  )
	                        +(edims3[0]  )*(edims3[1]  )*(edims3[2]+1);
	size_t numFaceGlobalOrdinalsPerProcess = 3* numElemPerProcess; // this doesn't count the duplicated shared faces
	
	if (numFacesPerProcess > std::numeric_limits<LocalIndex>::max() ) {
		std::cerr << "Error: numFacesPerProcess="<< numFacesPerProcess << " exceeds local ordinal max range " << std::numeric_limits<LocalIndex>::max() << std::endl;
		exit(1);
	}
	if (numFaceGlobalOrdinalsPerProcess*numProcesses > std::numeric_limits<GO>::max() ) {
		std::cerr << "Error: Requested global ordinals exceeds global ordinal max range." << std::endl;
		exit(1);
	}

	struct GO_PE_pair
	{
		GlobalOrdinal_t go;
		int PE;
		///GO_PE_pair() : go(-1), PE(-1) {}; //default constructor invalid values
		
		/**
		 * Group by PE, with one designated special PE being first grouping.
		 */
		struct customLess {
			
			// opreator< algorithm to establish GlobalOrdinal_t ordering
			// first by PE, where _specialPE sorts less than all other PE
			// then by GO
			bool operator()(const GO_PE_pair& a,const GO_PE_pair& b) const
			{
				if (a.PE == _specialPE && b.PE != _specialPE )
					return true;
				if (b.PE == _specialPE && a.PE != _specialPE )
					return false;
				//both special PE (equal) or neither special PE.
				if (a.PE > b.PE)
					return false;
				if (a.PE < b.PE)
					return true;
				// equal PE, now simple GlobalOrdinal_t ordering
				return (a.go < b.go);
			}
			customLess(int specialPE) : _specialPE(specialPE) {}
			int _specialPE;
			};
	};
	std::vector<GO_PE_pair> yourGO_PE_pair(numFacesPerProcess);
	
	// establish ordinal and connectivity for protoype based on a 3D box
	// omg.  burn this code after use.
	const GlobalOrdinal_t thisGOoffset = myrank*numFaceGlobalOrdinalsPerProcess; // yes elements, cause upper face is shared (same GO), no offset
	LocalIndex locount = 0;
	for (int facedim=0; facedim<3; facedim++) {
		for (int thisElemFlatIndex=0; thisElemFlatIndex<numElemPerProcess; thisElemFlatIndex++)
		{
			std::vector<int> myEposition = inverseRank(thisElemFlatIndex, edims3 );
			
			LocalIndex lo = localFaceOrdinal<LocalIndex>( facedim, myEposition, edims3 );
		
			// typical interior lower face in dim
			///myGlobalOrdinals[locount] = thisGOoffset + lo;
			yourGO_PE_pair[locount] = {thisGOoffset + lo, myrank}; // typical case most faces on same rank's local box
			///yourCorrespondingProcess[locount] = myrank; // typical case most faces on same ranks local box
			if (myEposition[facedim] == 0 ) // Correction needed: at lower extent in dim.
			{
				yourPposition = myPposition;
				///  Global Ordinal  is already set correctly
				yourPposition[facedim] = (myPposition[facedim]+pdims3[facedim] -1) % pdims3[facedim]; // periodic wrap-around at global edge
				// apply the correction:
				///yourCorrespondingProcess[locount] = flatIndex( yourPposition, pdims3 );
				yourGO_PE_pair[locount].PE = flatIndex( yourPposition, pdims3 );
			}
			locount++;
			
			if (myEposition[facedim] == (edims3[facedim]-1) ) // this elem. is at upper extent in facedim, which is "extra" face compared to element traverse
			{
				
				yourEposition = myEposition;
				yourPposition = myPposition;
				yourPposition[facedim] = (myPposition[facedim] +1) % pdims3[facedim]; // periodic wrap-around at global edge
				
				///yourCorrespondingProcess[locount] = flatIndex( yourPposition, pdims3 );
				yourGO_PE_pair[locount].PE = flatIndex( yourPposition, pdims3 );
				yourEposition[facedim] = 0; // at lower extent for that box
				LocalIndex yourFlatIndex = localFaceOrdinal<LocalIndex>( facedim, yourEposition, edims3 );
				///myGlobalOrdinals[locount] = yourCorrespondingProcess[locount]*numFaceGlobalOrdinalsPerProcess + yourFlatIndex;
				yourGO_PE_pair[locount].go = yourGO_PE_pair[locount].PE *numFaceGlobalOrdinalsPerProcess + yourFlatIndex;
				
				locount++; // an extra face.
			}
				

				// Is this duplicating the periodic face when extent==1 ?
				//  element extent can be one,  but that means two faces... two for me and two for you! ;-)
				// I think this is OK.
		}
		
	}
	assert(locount == numFacesPerProcess);
	// each variable element (geometrically a "face"), connects to one other process including self process
	
	std::sort(yourGO_PE_pair.begin(), yourGO_PE_pair.end(), GO_PE_pair::customLess(myrank) );
	
	// count number of partner processes, exchange their GlobalOrdinal_t ordering...
	int numNeighborPE = 0, lastNeighbor = myrank;
	LocalIndex localOffset = 0, lastOffset = 0, localExtent = 0;
	
	typedef typename OffsetMap_t::mapped_type LocalOffsetAndExtent;
	std::map<int, LocalOffsetAndExtent> neighborRankTo_LOCAL_OffsetAndExtent;
	for (const GO_PE_pair& i : yourGO_PE_pair) {
		if ( i.PE != lastNeighbor) {
			// insert characteristics of the previous GlobalOrdinal_t series
			neighborRankTo_LOCAL_OffsetAndExtent.insert( { lastNeighbor, { lastOffset, localExtent } } );
			numNeighborPE++;
			lastNeighbor = i.PE;
			lastOffset = localOffset;
		}
		localOffset++;
		localExtent = localOffset-lastOffset;
	}
	if (localOffset>0) // final end case.
		neighborRankTo_LOCAL_OffsetAndExtent.insert( { lastNeighbor, { lastOffset, localExtent  } } );
	// Decomposition Diagnostics:
	//std::cout << "P " << myrank << " has " << numNeighborPE << " neighbors." << std::endl;
	
	
	std::map<int, LocalOffsetAndExtent> neighborRankTo_REMOTE_OffsetAndExtent(neighborRankTo_LOCAL_OffsetAndExtent);
	
	//exchange localOffsets for each contiguous section of elements for a PE pair
	std::vector<MPI_Request> requests(2*numNeighborPE);
	int exchangeCounter=0;
	for (auto& i : neighborRankTo_REMOTE_OffsetAndExtent) {
		int pair_PE = i.first;
		if (pair_PE== myrank)
			continue;
		LocalIndex* recvBuf = &(i.second.offset);
		//*recvBuf = -1;
		const LocalIndex* sendBuf = &(neighborRankTo_LOCAL_OffsetAndExtent.at(pair_PE).offset);
		
		MPI_Sendrecv(sendBuf, 1, MPI_Datatypes<LocalIndex>::mpi_type(),
						 pair_PE, 999,
					 recvBuf, 1, MPI_Datatypes<LocalIndex>::mpi_type(),
					 pair_PE, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		//int ignoredRet1 =  MPI_Irecv( recvBuf, 1, MPI_Datatypes<LocalIndex>::mpi_type(), pair_PE, 0/*tag*/, MPI_COMM_WORLD, &requests[exchangeCounter++]);
		
		//int ignoredRet2 = MPI_Isend(sendBuf, 1, MPI_Datatypes<LocalIndex>::mpi_type(), pair_PE, 0/*tag*/, MPI_COMM_WORLD, &requests[exchangeCounter++]);
		
	}
	MPI_Waitall(exchangeCounter, &requests[0], MPI_STATUSES_IGNORE);
	
	// Decomposition Diagnostics:
	// std::stringstream ss;
	// ss << "PE " << myrank << ": pairs with " << std::endl;
	// for (const auto& i : neighborRankTo_REMOTE_OffsetAndExtent) {
	// 	int pair_PE = i.first;
	// 	ss << pair_PE << " (local, remote) Offset: ";
	// 	ss << "{ " << neighborRankTo_LOCAL_OffsetAndExtent.at(pair_PE).offset <<  ", " << i.second.offset << "}";
	// 	ss << "Extents: {" << neighborRankTo_LOCAL_OffsetAndExtent.at(pair_PE).extent <<  ", " << i.second.extent << "}"  << std::endl;
	// }
	// std::cout << ss.str();
	
	std::vector<GlobalOrdinal_t> myGlobalOrdinals;
	std::vector<int> yourCorrespondingProcess;
	// could be done with some iterator classes, but quick and dirty
	for (const GO_PE_pair& i : yourGO_PE_pair) {
		myGlobalOrdinals.push_back(i.go);
		yourCorrespondingProcess.push_back(i.PE);
	}
	
	// Decomposition Diagnostics:
	// for (int p = 0; p < numProcesses; p++)
	// {
	// 	if (p==myrank) {
	// 		std::cout << "P " << p << " Global Ordinals: " << makeCSLstr(myGlobalOrdinals) << std::endl;
	// 		std::cout << "P " << p << "	Your Process: " << makeCSLstr(yourCorrespondingProcess) << std::endl;
	// 		MPI_Barrier(MPI_COMM_WORLD);  // barrier actual *not* sequentially ordering stdout on Perlmutter / Slurm sys
	// 	}
	// }

	return std::make_tuple(myGlobalOrdinals, neighborRankTo_LOCAL_OffsetAndExtent, neighborRankTo_LOCAL_OffsetAndExtent);
}

/**
 * arg[1] - number of elements per process (default 8)
 * arg[2] - filename of shared library runtime module
 * arg[3] - specify kernel name to apply found in the runtime module
 */
int main(int argc, char *argv[]) {

	int numExpectedArguments = 3;
	int module_arg_adjust = 0;
#ifdef KERNEL_LINK_METHOD_RUNTIME_MODULE
	numExpectedArguments++;
	module_arg_adjust++;
#endif

    if (argc != numExpectedArguments) {
    std::cerr << "Exactly " << (numExpectedArguments-1) << " arguments expected:   numElements  "
#ifdef KERNEL_LINK_METHOD_RUNTIME_MODULE
	"runtimeModuleName.so  "
#endif
	"kernelName" <<std::endl;
    return 1;
    }
    
    int numElemPerProcess;
    {
        std::stringstream arg1(argv[1]);
        arg1 >> numElemPerProcess;
    }

#ifdef KERNEL_LINK_METHOD_RUNTIME_MODULE
    std::string runtime_module_filename;
    {
        std::stringstream arg2(argv[numExpectedArguments-2]);
        arg2 >> runtime_module_filename;
    }
#endif

    std::string kernel_name;
    {
        std::stringstream arg3(argv[numExpectedArguments-1+module_arg_adjust]);
        arg3 >> kernel_name;
    }

    int numranks, myrank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    debugger_attach_opportunity();

	setComputeDevice( myrank ); // bug on Perlmutter cray-mpich requires cpu-bind=none.  Then default placement is all ranks on same gpu 0.
	//std::cout << "PE " << myrank << ": GPU " << ComputeDevice::getGPUDeviceInfoString() << std::endl;

	typedef decltype(&get_kernel_by_name) get_kernel_by_name_fn_t;
	typedef decltype(&enqueueKernelWork_1D) enqueueKernelWork_fn_t;

#ifdef KERNEL_LINK_METHOD_RUNTIME_MODULE
    void * libhandle = dlopen(runtime_module_filename.c_str(), RTLD_NOW | RTLD_DEEPBIND ); //longterm preference: RTLD_LAZY);
    if (libhandle == nullptr) {
        std::cerr << "Error with dlopen " << runtime_module_filename.c_str() <<std::endl << dlerror() << std::endl;
        return 1;
    }

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

#elif defined(KERNEL_LINK_METHOD_COMPILE_TIME) || defined(KERNEL_LINK_METHOD_RTC)
	get_kernel_by_name_fn_t get_kernel_by_name_module1 = get_kernel_by_name;
	enqueueKernelWork_fn_t enqueueKernelWork_module1 = enqueueKernelWork_1D;

#else
#error "A KERNEL_LINK_METHOD_* macro is not defined."
#endif


	typedef DistributedVariable<double, LO, GO> DV_t;
	typedef NeighborhoodExchanger<DV_t, LO> NeighborhoodExchanger_t;
	typedef NeighborhoodExchanger_t::LocalOffsetAndExtent LocalOffsetAndExtent;

	typedef  std::map<int, LocalOffsetAndExtent> OffsetMap_t;
    
	// C++17 structured binding:
    const auto [ myGlobalOrdinals, neighborRankTo_LOCAL_OffsetAndExtent, neighborRankTo_REMOTE_OffsetAndExtent ] =
    createGlobalOrdinalDecomposition_3DstructuredFaces<LO, GO, OffsetMap_t>( numElemPerProcess, numranks, myrank );
	int num_faces_local = myGlobalOrdinals.size();
    
    DV_t myFaces( myGlobalOrdinals );
	DV_t yourFaces( myGlobalOrdinals );
	
	auto exchanger_p = new NeighborhoodExchanger_t ( myFaces, yourFaces, neighborRankTo_LOCAL_OffsetAndExtent, neighborRankTo_REMOTE_OffsetAndExtent );
	MPI_Barrier(MPI_COMM_WORLD);

    
    // Initialize the variable elements by GPU kernel
    int blockSize = 256;
    int numBlocks = (num_faces_local + blockSize - 1) / blockSize;

    void * my_faces_dev_ptr = myFaces.device_ptr();
	void * your_faces_dev_ptr = yourFaces.device_ptr();

    // Compute Device and stream(s)
    DeviceStream * pStream1 = getComputeDevice().createStream();

	Tracing::traceRangePush("kernels on stream1");
    // kernel run #1 : initialize elements based on runtime argument
    void * args1[] = {&num_faces_local, &my_faces_dev_ptr, &myrank};
    const KernelFn * user_choice_kernel = get_kernel_by_name_module1( kernel_name.c_str() );
    enqueueKernelWork_module1( pStream1, user_choice_kernel, numBlocks, blockSize, args1);
    
	// kernel run #2 : copy elements
    void * args2[] = {&num_faces_local, &my_faces_dev_ptr, &your_faces_dev_ptr};
    const KernelFn * copy_element_kernel = get_kernel_by_name_module1( "copy_element_kernel" );
    enqueueKernelWork_module1( pStream1, copy_element_kernel, numBlocks, blockSize, args2);

    std::vector<double> host_result(num_faces_local);
    
	pStream1->sync();
	Tracing::traceRangePop();

	// Initialization Diagnostics:
	// if (myrank==0) std::cout << "yourFaces Before Exchange:" << std::endl;
	// for (int p = 0; p < numranks; p++)
	// {
	// 	if (p==myrank) {
	// 		std::cout << "P " << p << ": " << makeCSLstr(host_result) << std::endl;
	// 		MPI_Barrier(MPI_COMM_WORLD);  // barrier actual *not* sequentially ordering stdout on Perlmutter / Slurm sys
	// 	}
	// }

	int numExchangeIterations=20;
	auto iterStart = std::chrono::high_resolution_clock::now();
	decltype(iterStart) firstIterStop;
	for (int count=0; count < numExchangeIterations; count++) {
		exchanger_p->exposureEpochBegin();
		exchanger_p->updateTargets();
		exchanger_p->exposureEpochEnd();
		if (count==0)
			firstIterStop = std::chrono::high_resolution_clock::now();
	}
	auto lastIterStop = std::chrono::high_resolution_clock::now();

	if (myrank==0 && numExchangeIterations>0) {
		std::cout << "Completed " << numExchangeIterations << " ghost face exchange iterations." << std::endl;
		long long first_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(firstIterStop - iterStart).count();
		long long remaining_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(lastIterStop-firstIterStop).count();

		std::cout << "    First epoch: " << first_microseconds << " microseconds" << std::endl;
		if (numExchangeIterations>1)
			std::cout << "    Remaining epochs: " << remaining_microseconds << " microseconds,  " << remaining_microseconds/double(numExchangeIterations-1) << " microseconds per iteration." << std::endl;		

	}
	pStream1->memcpy( &host_result[0], yourFaces.device_ptr(), num_faces_local*sizeof(double));
    pStream1->sync();
	
	// Output Diagnostics:
	// if (myrank==0) std::cout << "yourFaces After Exchange:" << std::endl;
	// for (int p = 0; p < numranks; p++)
	// {
	// 	if (p==myrank) {
	// 		std::cout << "P " << p << ": " << makeCSLstr(host_result) << std::endl;
	// 		MPI_Barrier(MPI_COMM_WORLD);  // barrier actual *not* sequentially ordering stdout on Perlmutter / Slurm sys
	// 	}
	// }

	getComputeDevice().freeStream(pStream1);
#ifdef KERNEL_LINK_METHOD_RUNTIME_MODULE
    int dlclose_results = dlclose(libhandle);
#endif

    return 0;
}

template <typename T>
std::string makeCSLstr( std::vector<T> v )
{
	std::stringstream ss;
	const char* sep = "";
	for (int p = 0; p < v.size(); p++)
	{
		ss << sep << v[p];
		sep=", ";
	}
	return ss.str();
}


void debugger_attach_opportunity()
{
    // usefull pause when debugging MPI runs based on environment variable
    int pauseID, myID;
    MPI_Comm_rank( /*(ompi_communicator_t*)*/ MPI_COMM_WORLD, &myID);
    volatile int holder = 0;
    if (const char* pPauseEnv = std::getenv("DEBUG_PAUSE_FOR_MPI_ID"))
    {
        std::stringstream ss( pPauseEnv );
        ss >> pauseID;
        if (pauseID == myID) {
            std::cerr << "Pausing to allow for debugger to attach to rank " << pauseID << " process id " << getpid() << "..." << std::endl;
            while ( holder==0)
                sleep(1);
        }
    }
}


