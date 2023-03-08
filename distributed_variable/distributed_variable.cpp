
#include "distributed_variable.h"
#include "compute_device_core_API.h"
#include <vector>
#include <assert.h>  // Define NDEBUG to disable
#include <stdexcept>
#include <map>
#include <algorithm>
#include <sstream>
#include <iostream> //cout

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


template <typename D>
struct MPI_Datatypes {
	static MPI_Datatype mpi_type();
};

template<> MPI_Datatype MPI_Datatypes<short>::mpi_type() { return MPI_SHORT;}
template<> MPI_Datatype MPI_Datatypes<int>::mpi_type() { return MPI_INT;}
template<> MPI_Datatype MPI_Datatypes<unsigned int>::mpi_type() { return MPI_UNSIGNED;}
template<> MPI_Datatype MPI_Datatypes<double>::mpi_type() { return MPI_DOUBLE;}
template<> MPI_Datatype MPI_Datatypes<float>::mpi_type() { return MPI_FLOAT;}



template <typename E, typename LocalIndex, typename GlobalOrdinal_t>
std::vector<GlobalOrdinal_t>
DistributedVariable<E, LocalIndex, GlobalOrdinal_t>::createGlobalOrdinalDecomposition_3DstructuredFaces(int numElemPerProcess, int numProcesses, int myrank )
{
	// Need to create a somewhat interesting arangement of 1:1 shared face relationships
	// implementing a 3D structured box
	std::vector<int> numRanksPrimeFactors =  primeFactors(numProcesses);
	std::vector<int> pdims3 = cubish( numRanksPrimeFactors );
	std::vector<int> numElementPrimeFactors =  primeFactors(numElemPerProcess);
	std::vector<int> edims3 = cubish( numElementPrimeFactors );
	std::vector<int> myPposition = inverseRank(myrank, pdims3);
	std::vector<int> yourPposition, yourEposition;
	
	//if (myrank==0) {
	//	std::cout << "Process Grid Dims: " << makeCSLstr(pdims3) << std::endl;
	//	std::cout << "Element Grid Dims (per process): " << makeCSLstr(edims3) << std::endl;
	//}
	
	// Working with FACES! not elements.  I'm prototyping a 3D box, so one extra face per dim...
	int numFacesPerProcess = (edims3[0]+1)*(edims3[1]  )*(edims3[2]  )
	                        +(edims3[0]  )*(edims3[1]+1)*(edims3[2]  )
	                        +(edims3[0]  )*(edims3[1]  )*(edims3[2]+1);
	int numFaceGlobalOrdinalsPerProcess = 3* numElemPerProcess; // this doesn't count the duplicated shared faces
	
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
			
			LocalIndex lo = localFaceOrdinal( facedim, myEposition, edims3 );
		
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
				LocalIndex yourFlatIndex = localFaceOrdinal( facedim, yourEposition, edims3 );
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
	struct LocalOffsetAndExtent {
		LocalIndex offset;
		LocalIndex extent;
	};
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
	std::cout << "P " << myrank << " has " << numNeighborPE << " neighbors." << std::endl;
	
	
	std::map<int, LocalOffsetAndExtent> neighborRankTo_REMOTE_OffsetAndExtent(neighborRankTo_LOCAL_OffsetAndExtent);
	
	//exchange localOffsets for each contiguous section of elements for a PE pair
	std::vector<MPI_Request> requests(2*numNeighborPE);
	int exchangeCounter=0;
	for (auto& i : neighborRankTo_REMOTE_OffsetAndExtent) {
		int pair_PE = i.first;
		if (pair_PE== myrank)
			continue;
		LocalIndex* recvBuf = &(i.second.offset);
		*recvBuf = -1;
		const LocalIndex* sendBuf = &(neighborRankTo_LOCAL_OffsetAndExtent.at(pair_PE).offset);
		
		MPI_Sendrecv(sendBuf, 1, MPI_Datatypes<LocalIndex>::mpi_type(),
						 pair_PE, 999,
					 recvBuf, 1, MPI_Datatypes<LocalIndex>::mpi_type(),
					 pair_PE, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		//int ignoredRet1 =  MPI_Irecv( recvBuf, 1, MPI_Datatypes<LocalIndex>::mpi_type(), pair_PE, 0/*tag*/, MPI_COMM_WORLD, &requests[exchangeCounter++]);
		
		//int ignoredRet2 = MPI_Isend(sendBuf, 1, MPI_Datatypes<LocalIndex>::mpi_type(), pair_PE, 0/*tag*/, MPI_COMM_WORLD, &requests[exchangeCounter++]);
		
	}
	MPI_Waitall(exchangeCounter, &requests[0], MPI_STATUSES_IGNORE);
	
	std::stringstream ss;
	ss << "PE " << myrank << ": pairs with " << std::endl;
	for (const auto& i : neighborRankTo_REMOTE_OffsetAndExtent) {
		int pair_PE = i.first;
		ss << pair_PE << " (local, remote) Offset: ";
		ss << "{ " << neighborRankTo_LOCAL_OffsetAndExtent.at(pair_PE).offset <<  ", " << i.second.offset << "}";
		ss << "Extents: {" << neighborRankTo_LOCAL_OffsetAndExtent.at(pair_PE).extent <<  ", " << i.second.extent << "}"  << std::endl;
	}
	std::cout << ss.str();
	
	std::vector<GlobalOrdinal_t> myGlobalOrdinals;
	std::vector<int> yourCorrespondingProcess;
	// could be done with some iterator classes, but quick and dirty
	for (const GO_PE_pair& i : yourGO_PE_pair) {
		myGlobalOrdinals.push_back(i.go);
		yourCorrespondingProcess.push_back(i.PE);
	}
}


// don't laugh. brain exercise. max range of interest ~1M.
std::vector<int> primeFactors(unsigned int n) {
	std::vector<int> factors;
	const int initial_n = n;
	if (n==1)
		return {1};
	int test=2;
	while (n > 1) {
		if (test>initial_n)
			throw std::runtime_error("Prime Factorization implemenation FAIL!");
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


template <typename E, typename LocalIndex, typename GlobalOrdinal_t>
LocalIndex
DistributedVariable<E, LocalIndex, GlobalOrdinal_t>::
localFaceOrdinal( int facedim, const std::vector<int> &multiIndex, const std::vector<int> &boxDim )
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
	assert(lo >=0 && lo < 3*boxDim[0]*boxDim[1]*boxDim[2]);
	return lo;
}

template <typename DV_t, typename OffsetMap_t>
NeighborhoodExchanger< DV_t, OffsetMap_t>::NeighborhoodExchanger( const DV_t& sourceVar, DV_t& targetVar, const OffsetMap_t& src_pe_offset_mapping, const OffsetMap_t& target_pe_offset_mapping )
: _src_pe_offset_mapping(src_pe_offset_mapping), _target_pe_offset_mapping(target_pe_offset_mapping)
{
	E * win_base = targetVar.device_ptr();
	int disp_unit = sizeof(E);
	MPI_Aint winsize = disp_unit * targetVar.numLocalElements();
	int res = MPI_Win_create(win_base, winsize, disp_unit, MPI_INFO_NULL /*consider: same_disp_unit*/, MPI_COMM_WORLD, &_mpi_win);

	
	//source
	_sourceBase = sourceVar.device_ptr();
	
	MPI_Group group_world;
	MPI_Comm_group(MPI_COMM_WORLD, &group_world);
	std::vector<int> targetPEvec;
	for (const auto& i : _target_pe_offset_mapping)
		targetPEvec.push_back(i.first);
	//fortunately every outbound is an inbound 1:1 so we only need one process group
	MPI_Group_incl(group_world, targetPEvec.size(), &targetPEvec[0], &_exchange_group);
	MPI_Group_free(&group_world);
	//starting from world to dwindle down to just a few ranks seems so inneficient. I don't see a better way in the API.
}
    
template <typename DV_t, typename OffsetMap_t>
NeighborhoodExchanger< DV_t, OffsetMap_t>::~NeighborhoodExchanger()
{
	MPI_Group_free(&_exchange_group);
	int res = MPI_Win_free(&_mpi_win);
}

/**
 * Initiates updates from source to target variable.
 *  Returns when source var is no longer in use.
 */
template <typename DV_t, typename OffsetMap_t>
void
NeighborhoodExchanger< DV_t, OffsetMap_t>::updateTargets()
{
	Tracing::traceRangePush(__FUNCTION__);
	MPI_Win_start(_exchange_group, 0/*assert default*/, _mpi_win); //carefully consider: MPI_MODE_NOCHECK
	
	// for each remote process and contiguous set of Elements, do a PUT
	for (const auto i : _target_pe_offset_mapping) {
		int targetPE = i.first;
		//old: int res1 = MPI_Win_lock(MPI_LOCK_SHARED, targetPE, /*assert*/ 0, _mpi_win);
		
		
		const E * origin_addr = _sourceBase + _src_pe_offset_mapping.at(targetPE).offset;
		int origin_count = i.second.extent; // same src/target count
		const MPI_Datatype dt = MPI_Datatypes<E>::mpi_type(); // same src/target datatype
		// TODO:  consider working in BYTES rather than any other type.  This will be most straightforward to extend to multi-node face exchange.
		
		int target_count = origin_count;
		MPI_Aint target_disp = i.second.offset;
		Tracing::traceRangePush("MPI_Put_innerLoop");
		int res2 = MPI_Put(origin_addr, origin_count, dt, targetPE, target_disp, target_count, dt, _mpi_win);
		Tracing::traceRangePop();
	
	}
	MPI_Win_complete(_mpi_win); // completes all PUTS at this origin, but not at the targets
	Tracing::traceRangePop();
}

/**
 * Begin exposure epoch for remote processes to modify local parts of targetVar.
 *  No local access to targetVar is allowable during exposure epoch
 */
template <typename DV_t, typename OffsetMap_t>
void
NeighborhoodExchanger< DV_t, OffsetMap_t>::exposureEpochBegin()
{
	Tracing::traceRangePush(__FUNCTION__);
	MPI_Win_post( _exchange_group, 0/*assert default*/, _mpi_win);
		//carefully consider: MPI_MODE_NOCHECK, MPI_MODE_NOSTORE
	Tracing::traceRangePop();
}

/**
 * End exposure epoch for remote processes to modify local parts of targetVar.
 *  Local access to targetVar is allowable outside of exposure epochs
 */
template <typename DV_t, typename OffsetMap_t>
void
NeighborhoodExchanger< DV_t, OffsetMap_t>::exposureEpochEnd()
{
	Tracing::traceRangePush(__FUNCTION__);
	MPI_Win_wait(_mpi_win);
	Tracing::traceRangePop();
}
