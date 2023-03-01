
/**
 * DistributedVariable of element types E.
 *  One-dimensional global and local element indexing.
 *  Local indexing is compact (zero to nLocalElements minus one)
 *  Local storage is contiguous.
 *
 */
 
#include <vector>


template <typename E, typename LocalIndex, typename GlobalOrdinal_t>
class DistributedVariable
{
public:
    
    DistributedVariable(const std::vector<GlobalOrdinal_t> &globalOrdinals )
    : _my_globalOrdinals(globalOrdinals), _device_storage(nullptr)
    {
        if (numLocalElements()>0)
        	_device_storage = static_cast<E*>(device_malloc(numLocalElements()*sizeof(E))); //TODO:, cudaMemAttachGlobal );
    }
    
    ~DistributedVariable()
    {
        if (numLocalElements()>0)
        	device_free( _device_storage );
    }
    
    /*  these are not simple for host access unless I establish unified address space
    E& operator(LocalIndex i);
    const E& operator(LocalIndex i) const;
     */
	
    size_t numLocalElements() const
    { return _my_globalOrdinals.size(); }
    
    E* device_ptr()
    { return _device_storage;}
	
private:
    DistributedVariable() = delete; // not needed since I provide non-default constructor
    
    std::vector<GlobalOrdinal_t> _my_globalOrdinals; // todo change to a unique id set
    E* _device_storage;
    
};

