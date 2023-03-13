
#include "distributed_variable.h"
#include "compute_device_core_API.h"
#include <vector>
#include <stdexcept>
#include <sstream>
#include <iostream> //cout

#include <cassert>



template <typename DV_t, typename LO>
NeighborhoodExchanger< DV_t, LO>::NeighborhoodExchanger( const DV_t& sourceVar, DV_t& targetVar, const OffsetMap_t& src_pe_offset_mapping, const OffsetMap_t& target_pe_offset_mapping )
: _src_pe_offset_mapping(src_pe_offset_mapping), _target_pe_offset_mapping(target_pe_offset_mapping)
{
	E * win_base = targetVar.device_ptr();
	int disp_unit = sizeof(E);
	MPI_Aint winsize = disp_unit * targetVar.numLocalElements();
	int res = MPI_Win_create(win_base, winsize, disp_unit, MPI_INFO_NULL /*consider: same_disp_unit*/, MPI_COMM_WORLD, &_mpi_win);
	assert(res==MPI_SUCCESS);
	
	//source
	_sourceBase = sourceVar.device_ptr();
	
	MPI_Group group_world;
	res = MPI_Comm_group(MPI_COMM_WORLD, &group_world); assert(res==MPI_SUCCESS);
	std::vector<int> targetPEvec;
	for (const auto& i : _target_pe_offset_mapping)
		targetPEvec.push_back(i.first);
	//fortunately every outbound is an inbound 1:1 so we only need one process group
	res = MPI_Group_incl(group_world, targetPEvec.size(), &targetPEvec[0], &_exchange_group); assert(res==MPI_SUCCESS);
	MPI_Group_free(&group_world);
	//starting from world to dwindle down to just a few ranks seems so inneficient. I don't see a better way in the API.

	//verbose diagnostics:
	//int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank); std::stringstream ss; ss <<"debug PE " << myrank << " created window size " << winsize << " of " << disp_unit << " bytes. ";
	//ss << "targetPEvecs: " << makeCSLstr(targetPEvec) << std::endl;
	//std::cout << ss.str();
}
    
template <typename DV_t, typename LO>
NeighborhoodExchanger< DV_t, LO>::~NeighborhoodExchanger()
{
	MPI_Group_free(&_exchange_group);
	int res = MPI_Win_free(&_mpi_win);
}

/**
 * Initiates updates from source to target variable.
 *  Returns when source var is no longer in use.
 */
template <typename DV_t, typename LO>
void
NeighborhoodExchanger< DV_t, LO>::updateTargets()
{
	Tracing::traceRangePush(__FUNCTION__);
	int res = MPI_Win_start(_exchange_group, 0/*assert default*/, _mpi_win); //carefully consider: MPI_MODE_NOCHECK
	assert(res==MPI_SUCCESS);
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
		//Diagnostic: int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank); std::cout << "debug PE " << myrank << ": MPI_Put to targetPE= " << targetPE << ", target_disp= " << target_disp << ", target_count= " << target_count << std::endl;
		int res2 = MPI_Put(origin_addr, origin_count, dt, targetPE, target_disp, target_count, dt, _mpi_win);
		assert(res2==MPI_SUCCESS);
		Tracing::traceRangePop();
	
	}
	MPI_Win_complete(_mpi_win); // completes all PUTS at this origin, but not at the targets
	Tracing::traceRangePop();
}

/**
 * Begin exposure epoch for remote processes to modify local parts of targetVar.
 *  No local access to targetVar is allowable during exposure epoch
 */
template <typename DV_t, typename LO>
void
NeighborhoodExchanger< DV_t, LO>::exposureEpochBegin()
{
	Tracing::traceRangePush(__FUNCTION__);
	int res = MPI_Win_post( _exchange_group, 0/*assert default*/, _mpi_win);
	assert(res==MPI_SUCCESS);
		//carefully consider: MPI_MODE_NOCHECK, MPI_MODE_NOSTORE
	Tracing::traceRangePop();
}

/**
 * End exposure epoch for remote processes to modify local parts of targetVar.
 *  Local access to targetVar is allowable outside of exposure epochs
 */
template <typename DV_t, typename LO>
void
NeighborhoodExchanger< DV_t, LO>::exposureEpochEnd()
{
	Tracing::traceRangePush(__FUNCTION__);
	int res = MPI_Win_wait(_mpi_win);
	assert(res==MPI_SUCCESS);
	Tracing::traceRangePop();
}



//bwah. template instantiation.
#ifndef LOCAL_ORDINAL_TYPE
#define LOCAL_ORDINAL_TYPE int
#endif
template class DistributedVariable<double, LOCAL_ORDINAL_TYPE, size_t>;
template class NeighborhoodExchanger<DistributedVariable<double, LOCAL_ORDINAL_TYPE, size_t>, LOCAL_ORDINAL_TYPE>;
