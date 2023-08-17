#pragma once

#include "devicehost.h"
#include "CSparseVector.h"
#include "VectorTypes.h"

//terminated CSR
template<typename T>
class CSparseMatrixCSR{
	int width;
	int height;
	T* values;//length is number of nonzeros	
	unsigned int* colIndices;//length is number of nonzeros
	unsigned int* rowStarts;//length is height+1(terminated CSR)
	int nonZeroCount;
public:
	//__device__ __host__ CSparseMatrixCSR(){}
	__device__ __host__ CSparseMatrixCSR(int width, int height,	T* values, unsigned int* colIndices,unsigned int* rowStarts, int nonZeroCount)
		:width(width),height(height),values(values),colIndices(colIndices),rowStarts(rowStarts),nonZeroCount(nonZeroCount){}

	__device__ __host__ int Width()const{return width;}	
	__device__ __host__ int Height()const{return height;}
	__device__ __host__ Int2 Size()const{return Int2(width,height);}
	__device__ __host__ int NonZeroCount()const{return nonZeroCount;}
	__device__ __host__ unsigned int* RowStarts(){return rowStarts;}
	__device__ __host__ unsigned int* ColIndices(){return colIndices;}
	__device__ __host__ T* Values(){return values;}

	__device__ __host__ int RowLength(int r)const{
		unsigned int rowStart=rowStarts[r];
		int rowLength=rowStarts[r+1]-rowStart;
		return rowLength;
	}

	__device__ __host__ unsigned int RowStart(int r)const{return rowStarts[r];}
	__device__ __host__ void GetRow(int r, T*& rowValues, unsigned int*& rowIndices, int& rowLength){
		unsigned int rowStart=rowStarts[r];
		rowLength=rowStarts[r+1]-rowStart;
		rowValues=values+rowStart;
		rowIndices=colIndices+rowStart;
	}	
	__device__ __host__ CSparseVector<T> GetRow(int r){
		unsigned int rowStart=rowStarts[r];
		int nonZeros=rowStarts[r+1]-rowStart;
		return CSparseVector<T>(values+rowStart,colIndices+rowStart,width,nonZeros);
	}
};