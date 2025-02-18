#pragma once
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "CudaCheckError.h"
#include "int64.h"
#include "Verify.h"

template<typename T>
class DeviceMemBlock{
	T* data;
public:
	DeviceMemBlock():data(0){}
	explicit DeviceMemBlock(int64 n){
		data=0;
		CudaCheckErrorImportant();
		cudaError_t e1=cudaMalloc(&data,n*sizeof(T));
//		cudaError_t e2=cudaMemset(data,0,n*sizeof(T));
		if(e1!=0){
			cudaGetLastError();//Needed to reset an error
			throw std::runtime_error("cudaMalloc failed. Out of GPU memory?");
		}
	}
	~DeviceMemBlock(){
		//Here we cannot throw exceptions, because this might be called during an exception
		cudaFree(data);
		data=0;
	}
	T* Pointer(){return (T*)data;}
};

	
