#pragma once

#include "DeviceVector.h"
#include "SparseDeviceVector.h"
#include "CSparseMatrixCSR.h"

//terminated CRS
template<typename T>
class SparseDeviceMatrixCSR{	
	int width;
	int height;
	DeviceVector<T> values;//length is number of nonzeros
	DeviceVector<unsigned int> colIndices;//length is number of nonzeros
	DeviceVector<unsigned int> rowStarts;//length is height+1(terminated CSR)
public:
	SparseDeviceMatrixCSR():width(0),height(0){}
	SparseDeviceMatrixCSR(int width, int height, DeviceVector<T> values,DeviceVector<unsigned int> colIndices,DeviceVector<unsigned int> rowStarts)
		:width(width),height(height),values(values),colIndices(colIndices),rowStarts(rowStarts){}

	int Width()const{return width;}
	int Height()const{return height;}
	int DimX()const{return width;}
	int DimY()const{return height;}
	Int2 Size()const{return Int2(width,height);}
	int64 NonZeroCount()const{return colIndices.Length();}
	DeviceVector<T> Values(){return values;}
	DeviceVector<unsigned int> ColIndices(){return colIndices;}
	DeviceVector<unsigned int> RowStarts(){return rowStarts;}
	CSparseMatrixCSR<T> GetC(){return CSparseMatrixCSR<T>(width,height,values.Data(),colIndices.Data(),rowStarts.Data(),(int)NonZeroCount());}
};

