
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
//#include "compute_device_core_API.h"
#include "simple_device_vector.h"

template <typename E, typename LocalIndex, typename GlobalOrdinal_t>
class DistributedVariable : public SimpleDeviceVector<E>
{
public:
    
    DistributedVariable(const std::vector<GlobalOrdinal_t> &globalOrdinals )
    : SimpleDeviceVector( globalOrdinals.size() ), _my_globalOrdinals(globalOrdinals)
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

typedef unsigned int GO;  typedef short LO;
template <typename D>
struct MPI_Datatypes {
    static MPI_Datatype mpi_type();
};

template<> MPI_Datatype MPI_Datatypes<short>::mpi_type() { return MPI_SHORT;}
template<> MPI_Datatype MPI_Datatypes<int>::mpi_type() { return MPI_INT;}
template<> MPI_Datatype MPI_Datatypes<unsigned int>::mpi_type() { return MPI_UNSIGNED;}
template<> MPI_Datatype MPI_Datatypes<double>::mpi_type() { return MPI_DOUBLE;}
template<> MPI_Datatype MPI_Datatypes<float>::mpi_type() { return MPI_FLOAT;}


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
    NeighborhoodExchanger( const DV_t& sourceVar, DV_t& targetVar, const OffsetMap_t& src_pe_offset_mapping, const OffsetMap_t& target_pe_offset_mapping )
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
    
    ~NeighborhoodExchanger()
    {
        MPI_Group_free(&_exchange_group);
        int res = MPI_Win_free(&_mpi_win);
    }

    /**
     * Initiates updates from source to target variable.
     *  Returns when source var is no longer in use.
     */
    void updateTargets()
    {
        nvtxRangePush(__FUNCTION__);
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
            nvtxRangePush("MPI_Put_innerLoop");
            int res2 = MPI_Put(origin_addr, origin_count, dt, targetPE, target_disp, target_count, dt, _mpi_win);
            nvtxRangePop();
        
        }
        MPI_Win_complete(_mpi_win); // completes all PUTS at this origin, but not at the targets
        nvtxRangePop();
    }
    
    /**
     * Begin exposure epoch for remote processes to modify local parts of targetVar.
     *  No local access to targetVar is allowable during exposure epoch
     */
    void exposureEpochBegin()
    {
        nvtxRangePush(__FUNCTION__);
        MPI_Win_post( _exchange_group, 0/*assert default*/, _mpi_win);
          //carefully consider: MPI_MODE_NOCHECK, MPI_MODE_NOSTORE
        nvtxRangePop();
    }
    
    /**
     * End exposure epoch for remote processes to modify local parts of targetVar.
     *  Local access to targetVar is allowable outside of exposure epochs
     */
    void exposureEpochEnd()
    {
        nvtxRangePush(__FUNCTION__);
        MPI_Win_wait(_mpi_win);
        nvtxRangePop();
    }
    
    OffsetMap_t _src_pe_offset_mapping, _target_pe_offset_mapping;
    MPI_Win _mpi_win;
    MPI_Group _exchange_group;
    const E * _sourceBase;
};


#endif //DISTRIBUTED_VARIABLE_H
