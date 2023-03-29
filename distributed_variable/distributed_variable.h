
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
#include <map>
#include <mpi.h>

//#include "compute_device_core_API.h"
#include "simple_device_vector.h"



template <typename D>
struct MPI_Datatypes {
	static MPI_Datatype mpi_type();
};




template <typename E, typename LocalIndex, typename GlobalOrdinal_t>
class DistributedVariable : public SimpleDeviceVector<E>
{
	public:

	typedef E ElementType;

	DistributedVariable(const std::vector<GlobalOrdinal_t> &globalOrdinals )
	: SimpleDeviceVector<E>( globalOrdinals.size() ), _my_globalOrdinals(globalOrdinals)
	{

	}

	~DistributedVariable()
	{

	}

	size_t numLocalElements() const
	{ return _my_globalOrdinals.size(); }

	

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
template <typename DV_t, typename LO>
class NeighborhoodExchanger
{
	public:
	typedef typename DV_t::ElementType E;
	struct LocalOffsetAndExtent {
		LO offset;
		LO extent;
	};
	typedef std::map<int, LocalOffsetAndExtent> OffsetMap_t;


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
