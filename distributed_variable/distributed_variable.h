
/**
 * DistributedVariable of element types E.
 *  One-dimensional global and local element indexing.
 *  Local indexing is compact (zero to nLocalElements minus one)
 *  Local storage is contiguous.
 *
 */

#ifndef DISTRIBUTED_VARIABLE_H
#define DISTRIBUTED_VARIABLE_H


#include <vector>

#include <mpi.h>

//#include "compute_device_core_API.h"
#include "simple_device_vector.h"

template <typename E, typename LocalIndex, typename GlobalOrdinal_t>
class DistributedVariable : public SimpleDeviceVector<E>
{
	public:

	DistributedVariable(const std::vector<GlobalOrdinal_t> &globalOrdinals )
	: SimpleDeviceVector<E>( globalOrdinals.size() ), _my_globalOrdinals(globalOrdinals)
	{

	}

	~DistributedVariable()
	{

	}

	/**
	 * Calculate local ordinal for a face normal to facedim.
	 *  Ordering is all faces normal to dim0, then dim1, then dim2.  C-style row-major ordering in those groups
	 *  "local" because upper extent faces on the box are not counted
	 */
	LocalIndex localFaceOrdinal( int facedim, const std::vector<int> &multiIndex, const std::vector<int> &boxDim );


	size_t numLocalElements() const
	{ return _my_globalOrdinals.size(); }

	static std::vector<GlobalOrdinal_t> createGlobalOrdinalDecomposition_3DstructuredFaces(int numElemPerProcess, int numProcesses, int myrank );

	private:
	DistributedVariable() = delete; // not needed since I provide non-default constructor

	std::vector<GlobalOrdinal_t> _my_globalOrdinals; // todo change to a unique id set

};



/**
 * Neighborhood exchange update helper for DistributedVariable pairs
 *  Copy element values from source to target variable assuming same
 *  global ordinal ordering and distribution.
 *
 *  Expected use sequence:
 *   1. exposureEpochBegin()
 *   2. updateTargets()
 *   3. exposureEpochEnd()
 */
template <typename DV_t, typename OffsetMap_t>
class NeighborhoodExchanger
{
	public:
	typedef typename DV_t::ElementType E;
	NeighborhoodExchanger( const DV_t& sourceVar, DV_t& targetVar, const OffsetMap_t& src_pe_offset_mapping, const OffsetMap_t& target_pe_offset_mapping );

	~NeighborhoodExchanger();

	/**
	 * Initiates updates from source to target variable.
	 *  Returns when source var is no longer in use.
	 */
	void updateTargets();

	/**
	 * Begin exposure epoch for remote processes to modify local parts of targetVar.
	 *  No local access to targetVar is allowable during exposure epoch
	 */
	void exposureEpochBegin();

	/**
	 * End exposure epoch for remote processes to modify local parts of targetVar.
	 *  Local access to targetVar is allowable outside of exposure epochs
	 */
	void exposureEpochEnd();

	OffsetMap_t _src_pe_offset_mapping, _target_pe_offset_mapping;

	/*TODO:  move these to an impl class if they're the only remnants of mpi.h in this header*/
	MPI_Win _mpi_win;
	MPI_Group _exchange_group;
	const E * _sourceBase;
};


#endif //DISTRIBUTED_VARIABLE_H
