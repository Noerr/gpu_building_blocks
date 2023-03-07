
/**
 * Simple vector of element types E.
 *  One-dimensional global and local element indexing.
 *  Local indexing is compact (zero to nLocalElements minus one)
 *  Local storage is contiguous.
 *
 */
 
#ifndef SIMPLE_DEVICE_VECTOR_H
#define SIMPLE_DEVICE_VECTOR_H

template <typename E>
class SimpleDeviceVector
{
public:
    
    SimpleDeviceVector(size_t numElements )
    : _numElements(numElements), _device_storage(nullptr)
    {
        if (size()>0)
        	_device_storage = static_cast<E*>(getComputeDevice().malloc(size()*sizeof(E)));
    }
    
    ~SimpleDeviceVector()
    {
        if (size()>0)
        	getComputeDevice().free( _device_storage );
    }
    

    size_t size() const
    { return _numElements; }
    
    E* device_ptr()
    { return _device_storage;}
	
private:
    SimpleDeviceVector() = delete; // not needed since I provide non-default constructor
    
    size_t _numElements; // todo change to a unique id set
    E* _device_storage;
    
};

#endif //SIMPLE_DEVICE_VECTOR_H
