#pragma once

#include "devicehost.h"

template<typename T>
class CSparseVector{
public:	
	T* values;	
	unsigned int* indices;
	int length;
	int nonZeroCount;
public:
	__device__ __host__ CSparseVector(T* values, unsigned int* indices, int length, int nonZeroCount):values(values),indices(indices),length(length),nonZeroCount(nonZeroCount){}

	
	__device__ __host__ int Length()const{return length;}
	__device__ __host__ int DimX()const{return length;}
	__device__ __host__ int NonZeroCount()const{return nonZeroCount;}
	__device__ __host__ T* Values(){return values;}
	__device__ __host__ unsigned int* Indices(){return indices;}
	__device__ __host__ T& Value(int i){return values[i];}
	__device__ __host__ const T& Value(int i)const{return values[i];}
	__device__ __host__ unsigned int& Index(int i){return indices[i];}
	__device__ __host__ unsigned int Index(int i)const{return indices[i];}
};