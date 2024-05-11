#pragma once

#include "ldg.h"
#include "CSparseMatrixCSR.h"
#include "CSparseVector.h"

//Multiply sparse vector with sparse matrix using a warp of threads.
//Merge up to WarpSize rows. Each thread pulls from his row. 
//rowA must have at most WarpSize elements
//Result (dst) must be pre-allocated.
template<int WarpSize, bool AssumeOnes, typename T>
static __device__ void MulWarp(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B,int thread){
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=AssumeOnes?1.0:a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=thread;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=AssumeOnes?b.Value(i):weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues;uint* rowIndices;int rowLength=0;//The row for the thread
	T weight=0;//The weight for the row
	if(thread<a.NonZeroCount()){
		uint r=ldg(a.Indices()+thread);//uint rowIndex=a.Index(thread);		
		uint rowStart=ldg(B.RowStarts()+r);
		rowLength=ldg(B.RowStarts()+r+1)-rowStart;
		rowValues=B.Values()+rowStart;
		rowIndices=B.ColIndices()+rowStart;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight=AssumeOnes?1.0:ldg(a.Values()+thread);//a.Value(thread);
	}

	int rowPos=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread
	if(rowPos<rowLength){//Load the front index and row
		frontIndex=ldg(rowIndices+rowPos);//ldg: explicit cache usage
		frontValue=AssumeOnes?ldg(rowValues+rowPos):ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos<rowLength){
				frontValue=AssumeOnes?ldg(rowValues+rowPos):ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
				frontIndex=(int)ldg(rowIndices+rowPos);//ldg: explicit cache usage
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		T sum=WarpSum<WarpSize>(tmp);
		if(thread==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && thread<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+thread]=bufferedIndex;
			dst.Values()[dstPos+thread]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}


//Multiply sparse vector with sparse matrix using a warp of threads.
//Merge up to WarpSize rows. Each thread pulls from his row. 
//rowA must have at most WarpSize elements
//Result (dst) must be pre-allocated.
template<int WarpSize, typename T>
static __device__ void DifMul_1(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B){
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);

		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues;uint* rowIndices;int rowLength=0;//The row for the thread
	uint weight=0;//The weight for the row
	if(threadIdx.x<a.NonZeroCount()){
		uint r=ldg(a.Indices()+threadIdx.x);//uint rowIndex=a.Index(thread);		
		uint rowStart=ldg(B.RowStarts()+r);
		rowLength=ldg(B.RowStarts()+r+1)-rowStart;
		rowValues=B.Values()+rowStart;
		rowIndices=B.ColIndices()+rowStart;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight=ldg(a.Values()+threadIdx.x);//a.Value(thread);
	}

	int rowPos=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue = T{};//the front of the row of the thread
	if(rowPos<rowLength){//Load the front index and row
		frontIndex=ldg(rowIndices+rowPos);//ldg: explicit cache usage
		frontValue=ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp= T{};//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos<rowLength){
				frontValue=ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
				frontIndex=(int)ldg(rowIndices+rowPos);//ldg: explicit cache usage
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ uint MulOverWarp(CSparseVector<T>& dst_temp, CSparseVector<T>& a, CSparseMatrixCSR<T>& B){
	if(a.NonZeroCount()==0)//nothing to do
		return 0;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst_temp.NonZeroCount();i+=WarpSize){
			dst_temp.Index(i)=b.Index(i);
			dst_temp.Value(i)=weight*b.Value(i);
		}
		return dst_temp.nonZeroCount;
	}
 
	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;
	int segmentId = warpId%SegmentSize;

	int c_rowLength=0;

	int ctemp_segmentSize=dst_temp.nonZeroCount/(SegmentSize*2);

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues;uint* rowIndices;int rowLength=0;//The row for the thread
	T weight=0;//The weight for the row
	if(threadIdx.x<a.NonZeroCount()){
		uint r=ldg(a.Indices()+threadIdx.x);//uint rowIndex=a.Index(thread);		
		uint rowStart=ldg(B.RowStarts()+r);
		rowLength=ldg(B.RowStarts()+r+1)-rowStart;
//		rowLength = 32;
		rowValues=B.Values()+rowStart;
		rowIndices=B.ColIndices()+rowStart;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight=ldg(a.Values()+threadIdx.x);//a.Value(thread);
	}

	int rowPos=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread
	if(rowPos<rowLength){//Load the front index and row
		frontIndex=ldg(rowIndices+rowPos);//ldg: explicit cache usage
		frontValue=ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	uint rowLength2=0;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos<rowLength){
				frontValue=ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
				frontIndex=(int)ldg(rowIndices+rowPos);//ldg: explicit cache usage
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}

		minFront=WarpMin<WarpSize>(frontIndex);

//		printf("bufferPos = %d\n", bufferPos);
//		printf("in OverMulWarp. threadIdx.x=%d, threadIdx.y=%d, bufferedIndex = %d, bufferedValue = %f\n", threadIdx.x, threadIdx.y, bufferedIndex, bufferedValue);

		bufferPos++;		
		rowLength2++;
		if(bufferPos==WarpSize || (minFront==intMax && laneId<bufferPos)){//Save buffer to global memory (coalesced)
			dst_temp.Indices()[segmentId*ctemp_segmentSize+dstPos+laneId]=bufferedIndex;
			dst_temp.Values()[segmentId*ctemp_segmentSize+dstPos+laneId]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}



	if(rowLength2<ctemp_segmentSize)
	{
		if(laneId==0)
		{
			int rowStart_last = rowLength2;
			dst_temp.Indices()[segmentId*ctemp_segmentSize+rowStart_last] = intMax;
		}
	}

//	if(laneId==0)
//	{
//		rowLength_temp[warpId] = rowLength2;
//	}
	__syncthreads();

//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0 && threadIdx.y==0)
//		{
//			uint indices;
//			T value;
//			for(int k=0; k<66; k++)
//			{
//				indices = dst_temp.Indices()[k];
//				value = dst_temp.Values()[k];
//				printf("dst_temp.Indices[%d] = %d, dst_temp.Values[%d] = %f\n",k, indices, k, value);
//			}
//
//		}
//	}

	if(segmentId==0)
	{
		rowLength=0;
		if(threadIdx.x<SegmentSize){
			uint rowStart=threadIdx.x*ctemp_segmentSize;
//			rowLength=rowLength_temp[laneId+(warpId/c_segmentSize)*c_segmentSize];
			rowLength=ctemp_segmentSize;
			rowValues=dst_temp.Values()+rowStart;
			rowIndices=dst_temp.Indices()+rowStart;
			//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		}
//		__syncthreads();

		rowPos=0;//Current position into row
		frontIndex=intMax;//The front index of the row. intMax means that the row ended.
		frontValue=0;//the front of the row of the thread
		if(rowPos<rowLength){//Load the front index and row
			frontIndex=ldg(rowIndices+rowPos);//ldg: explicit cache usage
			frontValue=ldg(rowValues+rowPos);//ldg: explicit cache usage
			rowPos++;
		}

		 minFront=WarpMin<SegmentSize>(frontIndex);//The smallest index
		 dstPos=0;

		//Results are stored into a "buffer" of registers.
		//When WarpSize results are available, the buffer is saved to global mem (coalesced)
		bufferPos=0;//how many elements are in the buffer
		while(minFront!=intMax){//Compute one element per iteration
			T tmp=0.0;//Used to compute the value
			if(frontIndex==minFront){//put these into tmp and load next elements
				tmp=frontValue;
				//load next
				if(rowPos<rowLength){
					frontValue=ldg(rowValues+rowPos);//ldg: explicit cache usage
					frontIndex=(int)ldg(rowIndices+rowPos);//ldg: explicit cache usage
					rowPos++;
				}
				else//out of the game
					frontIndex=intMax;
			}
			T sum=WarpSum<SegmentSize>(tmp);

			if(threadIdx.x==bufferPos){//Save into buffer
				bufferedIndex=(uint)minFront;
				bufferedValue=sum;
			}

			minFront=WarpMin<SegmentSize>(frontIndex);

			c_rowLength++;
			bufferPos++;		
			if(bufferPos==SegmentSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
				dst_temp.Indices()[(dst_temp.nonZeroCount/2)+dstPos+threadIdx.x]=bufferedIndex;
				dst_temp.Values()[(dst_temp.nonZeroCount/2)+dstPos+threadIdx.x]=bufferedValue;
				dstPos+=SegmentSize;
				bufferPos=0;
			}		
		}

	}

	return c_rowLength;
}

template<int c_segmentSize,typename T>
static __device__ void MulOverCopy(CSparseVector<T>& dst, CSparseVector<T> &dst_temp, uint rowLength)
{
	if(dst_temp.nonZeroCount==0)
		return;

	uint loop=DivUp(rowLength,(uint)blockDim.x);

	for(int i=0; i<loop;i++)
	{
	//		if(blockIdx.x==0)
	//		{
	//			if(threadIdx.x==0)
	//			{
	//				printf("In for loop\n");
	//			}
	//		}
		if((threadIdx.x+i*blockDim.x)<rowLength)
		{
			dst.Indices()[threadIdx.x+i*blockDim.x] = dst_temp.Indices()[(dst_temp.nonZeroCount/2)+threadIdx.x+i*blockDim.x];
			dst.Values()[threadIdx.x+i*blockDim.x] = dst_temp.Values()[(dst_temp.nonZeroCount/2)+threadIdx.x+i*blockDim.x];
		}
	}
//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0)
//		{
//			printf("loop=%d\n",loop);
//			printf("dst_temp.nonZeroCount=%d\n",dst_temp.nonZeroCount);
//			printf("rowLength=%d\n",rowLength);
//			printf("dst.Indices()[0]=%d\n",dst.Indices()[0]);
//		    printf("dst_temp.Indices()[0] =%d\n",dst_temp.Indices()[0]);
//		}
//	}
}

template<typename T>
static __device__ void MulCopy(CSparseVector<T>& dst, CSparseVector<T> &dst_temp)
{
	if(dst_temp.nonZeroCount==0)
		return;

	uint loop=DivUp((uint)dst_temp.nonZeroCount,(uint)blockDim.x);

	for(int i=0; i<loop;i++)
	{
		if((threadIdx.x+i*blockDim.x)<dst_temp.nonZeroCount)
		{
		//	if(blockIdx.x==0)
		//	{
		//		if(threadIdx.x==0)
		//		{
		//			printf("In for loop\n");
		//		}
		//	}
			dst.Indices()[threadIdx.x+i*blockDim.x] = dst_temp.Indices()[threadIdx.x+i*blockDim.x];
			dst.Values()[threadIdx.x+i*blockDim.x] = dst_temp.Values()[threadIdx.x+i*blockDim.x];
		}
	}

//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0)
//		{
//			printf("loop=%d\n",loop);
//			printf("dst_temp.nonZeroCount=%d\n",dst_temp.nonZeroCount);
//			printf("dst.Indices()[0]=%d\n",dst.Indices()[0]);
//		    printf("dst_temp.Indices()[0] =%d\n",dst_temp.Indices()[0]);
//		}
//	}
}

//Ni-to-M(Ni: the number of nonzeros of Ai; M:the number of threads)
template<int WarpSize, typename T>
static __device__ void DifMul_2(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B){
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	int t=(threadIdx.x+1)*2;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*2;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount()){

		uint d0=threadIdx.x*2;
		uint r0=ldg(a.Indices()+d0);
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowLength1=0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}

	if(index0<index1)
	{
		frontIndex=index0;
		frontValue=ldg(rowValues0+rowPos0)*weight0;
		rowPos0++;
	}
	else if(index0>index1)
	{
		frontIndex=index1;
		frontValue=ldg(rowValues1+rowPos1)*weight1;
		rowPos1++;
	}
	else
	{
		if(index0!=intMax)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0+ldg(rowValues1+rowPos1)*weight1;
			rowPos0++;
			rowPos1++;
		}
		else
		{
		}
	}


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(index0<index1)
			{
				frontIndex=index0;
				frontValue=ldg(rowValues0+rowPos0)*weight0;
				rowPos0++;
			}
			else if(index0>index1)
			{
				frontIndex=index1;
				frontValue=ldg(rowValues1+rowPos1)*weight1;
				rowPos1++;
			}
			else 
			{
				if(index0!=intMax)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0+ldg(rowValues1+rowPos1)*weight1;
					rowPos0++;
					rowPos1++;
				}
				else
				{
					frontIndex=intMax;
				}
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, typename T>
static __device__ void DifMul_4(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B){
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	int t=(threadIdx.x+1)*4;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*4;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%4==3
	{
		uint d0=threadIdx.x*4;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}
	else if(t-2==a.NonZeroCount()) //a.NonZeroCount()%4==2
	{
		uint d0=threadIdx.x*4;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-3==a.NonZeroCount()) //a.NonZeroCount()%4==1
	{
		uint d0=threadIdx.x*4;
		uint r0=ldg(a.Indices()+d0);
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);

	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
	}
	else
	{
		frontIndex=intMax;
	}


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, typename T>
static __device__ void DifMul_8(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B){
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	int t=(threadIdx.x+1)*8;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%8==7
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
	}
	else if(t-2==a.NonZeroCount()) //a.NonZeroCount()%8==6
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint r5=ldg(a.Indices()+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
	}
	else if(t-3==a.NonZeroCount())// a.NonZeroCount()%8==5
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
	}
	else if(t-4==a.NonZeroCount())// a.NonZeroCount()%8==4
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}
	else if(t-5==a.NonZeroCount())// a.NonZeroCount()%8==3
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}
	else if(t-6==a.NonZeroCount())// a.NonZeroCount()%8==2
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-7==a.NonZeroCount())// a.NonZeroCount()%8==1
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	frontIndex=min_index;

	
	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=ldg(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=ldg(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=ldg(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=ldg(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
	}
	else
	{
		frontIndex=intMax;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=ldg(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=ldg(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=ldg(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=ldg(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, typename T>
static __device__ void DifMul_16(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B){
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread
	T* rowValues8;uint* rowIndices8;int rowLength8=0;//The row for the thread
	T* rowValues9;uint* rowIndices9;int rowLength9=0;//The row for the thread
	T* rowValues10;uint* rowIndices10;int rowLength10=0;//The row for the thread
	T* rowValues11;uint* rowIndices11;int rowLength11=0;//The row for the thread
	T* rowValues12;uint* rowIndices12;int rowLength12=0;//The row for the thread
	T* rowValues13;uint* rowIndices13;int rowLength13=0;//The row for the thread
	T* rowValues14;uint* rowIndices14;int rowLength14=0;//The row for the thread
	T* rowValues15;uint* rowIndices15;int rowLength15=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	T weight8=0;//The weight for the row
	T weight9=0;//The weight for the row
	T weight10=0;//The weight for the row
	T weight11=0;//The weight for the row
	T weight12=0;//The weight for the row
	T weight13=0;//The weight for the row
	T weight14=0;//The weight for the row
	T weight15=0;//The weight for the row
	int t=(threadIdx.x+1)*16;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%16==15
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
	}
	else if(t-2==a.NonZeroCount()) //a.NonZeroCount()%16==14
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
	}
	else if(t-3==a.NonZeroCount())// a.NonZeroCount()%16==13
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
	}
	else if(t-4==a.NonZeroCount())// a.NonZeroCount()%16==12
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
	}
	else if(t-5==a.NonZeroCount())// a.NonZeroCount()%16==11
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
	}
	else if(t-6==a.NonZeroCount())// a.NonZeroCount()%16==10
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
	}
	else if(t-7==a.NonZeroCount())// a.NonZeroCount()%16==9
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
	}
	else if(t-8==a.NonZeroCount())// a.NonZeroCount()%16==8
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
	}
	else if(t-9==a.NonZeroCount())// a.NonZeroCount()%16==7
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
	}
	else if(t-10==a.NonZeroCount())// a.NonZeroCount()%16==6
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
	}
	else if(t-11==a.NonZeroCount())// a.NonZeroCount()%16==5
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
	}
	else if(t-12==a.NonZeroCount())// a.NonZeroCount()%16==4
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}
	else if(t-13==a.NonZeroCount())// a.NonZeroCount()%16==3
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}
	else if(t-14==a.NonZeroCount())// a.NonZeroCount()%16==2
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-15==a.NonZeroCount())// a.NonZeroCount()%16==1
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int rowPos8=0;//Current position into row
	int rowPos9=0;//Current position into row
	int rowPos10=0;//Current position into row
	int rowPos11=0;//Current position into row
	int rowPos12=0;//Current position into row
	int rowPos13=0;//Current position into row
	int rowPos14=0;//Current position into row
	int rowPos15=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	int index8=intMax;
	int index9=intMax;
	int index10=intMax;
	int index11=intMax;
	int index12=intMax;
	int index13=intMax;
	int index14=intMax;
	int index15=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}
	if(rowPos8<rowLength8){
		index8=ldg(rowIndices8+rowPos8);
	}
	if(rowPos9<rowLength9){
		index9=ldg(rowIndices9+rowPos9);
	}
	if(rowPos10<rowLength10){
		index10=ldg(rowIndices10+rowPos10);
	}
	if(rowPos11<rowLength11){
		index11=ldg(rowIndices11+rowPos11);
	}
	if(rowPos12<rowLength12){
		index12=ldg(rowIndices12+rowPos12);
	}
	if(rowPos13<rowLength13){
		index13=ldg(rowIndices13+rowPos13);
	}
	if(rowPos14<rowLength14){
		index14=ldg(rowIndices14+rowPos14);
	}
	if(rowPos15<rowLength15){
		index15=ldg(rowIndices15+rowPos15);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	min_index=index8<min_index?index8:min_index;
	min_index=index9<min_index?index9:min_index;
	min_index=index10<min_index?index10:min_index;
	min_index=index11<min_index?index11:min_index;
	min_index=index12<min_index?index12:min_index;
	min_index=index13<min_index?index13:min_index;
	min_index=index14<min_index?index14:min_index;
	min_index=index15<min_index?index15:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=ldg(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=ldg(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=ldg(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=ldg(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
		if(index8==min_index)
		{
			frontValue+=ldg(rowValues8+rowPos8)*weight8;
			rowPos8++;
		}
		if(index9==min_index)
		{
			frontValue+=ldg(rowValues9+rowPos9)*weight9;
			rowPos9++;
		}
		if(index10==min_index)
		{
			frontValue+=ldg(rowValues10+rowPos10)*weight10;
			rowPos10++;
		}
		if(index11==min_index)
		{
			frontValue+=ldg(rowValues11+rowPos11)*weight11;
			rowPos11++;
		}
		if(index12==min_index)
		{
			frontValue+=ldg(rowValues12+rowPos12)*weight12;
			rowPos12++;
		}
		if(index13==min_index)
		{
			frontValue+=ldg(rowValues13+rowPos13)*weight13;
			rowPos13++;
		}
		if(index14==min_index)
		{
			frontValue+=ldg(rowValues14+rowPos14)*weight14;
			rowPos14++;
		}
		if(index15==min_index)
		{
			frontValue+=ldg(rowValues15+rowPos15)*weight15;
			rowPos15++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
	//		frontIndex=index0>index1?index1:index0;
	//		frontValue=index0>index1?ldg(rowValues1+rowPos1)*weight1:ldg(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//	if(threadIdx.x==1&&threadIdx.y==0)
	//	{
	//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
	//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
	//		printf("weight0=%f,weight1=%f,weight2=%f,weight3=%f,weight4=%f,weight5=%f,weight6=%f,weight7=%f\n",weight0,weight1,weight2,weight3,weight4,weight5,weight6,weight7);
	//		printf("weight8=%f,weight9=%f,weight10=%f,weight11=%f,weight12=%f,weight13=%f,weight14=%f,weight15=%f\n",weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15);
	//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
	//		printf("minFront=%d\n",minFront);
	//		printf("------------------------------------\n");
	//	}
	//	if(threadIdx.x==0&&threadIdx.y==0)
	//	{
	//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
	//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
	//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
	//		printf("minFront=%d\n",minFront);
	//		printf("------------------------------------\n");
	//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}
			if(rowPos8<rowLength8){
				index8=ldg(rowIndices8+rowPos8);
			}
			else{
				index8=intMax;
			}
			if(rowPos9<rowLength9){
				index9=ldg(rowIndices9+rowPos9);
			}
			else{
				index9=intMax;
			}
			if(rowPos10<rowLength10){
				index10=ldg(rowIndices10+rowPos10);
			}
			else{
				index10=intMax;
			}
			if(rowPos11<rowLength11){
				index11=ldg(rowIndices11+rowPos11);
			}
			else{
				index11=intMax;
			}
			if(rowPos12<rowLength12){
				index12=ldg(rowIndices12+rowPos12);
			}
			else{
				index12=intMax;
			}
			if(rowPos13<rowLength13){
				index13=ldg(rowIndices13+rowPos13);
			}
			else{
				index13=intMax;
			}
			if(rowPos14<rowLength14){
				index14=ldg(rowIndices14+rowPos14);
			}
			else{
				index14=intMax;
			}
			if(rowPos15<rowLength15){
				index15=ldg(rowIndices15+rowPos15);
			}
			else{
				index15=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			min_index=index8<min_index?index8:min_index;
			min_index=index9<min_index?index9:min_index;
			min_index=index10<min_index?index10:min_index;
			min_index=index11<min_index?index11:min_index;
			min_index=index12<min_index?index12:min_index;
			min_index=index13<min_index?index13:min_index;
			min_index=index14<min_index?index14:min_index;
			min_index=index15<min_index?index15:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=ldg(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=ldg(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=ldg(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=ldg(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
				if(index8==min_index)
				{
					frontValue+=ldg(rowValues8+rowPos8)*weight8;
					rowPos8++;
				}
				if(index9==min_index)
				{
					frontValue+=ldg(rowValues9+rowPos9)*weight9;
					rowPos9++;
				}
				if(index10==min_index)
				{
					frontValue+=ldg(rowValues10+rowPos10)*weight10;
					rowPos10++;
				}
				if(index11==min_index)
				{
					frontValue+=ldg(rowValues11+rowPos11)*weight11;
					rowPos11++;
				}
				if(index12==min_index)
				{
					frontValue+=ldg(rowValues12+rowPos12)*weight12;
					rowPos12++;
				}
				if(index13==min_index)
				{
					frontValue+=ldg(rowValues13+rowPos13)*weight13;
					rowPos13++;
				}
				if(index14==min_index)
				{
					frontValue+=ldg(rowValues14+rowPos14)*weight14;
					rowPos14++;
				}
				if(index15==min_index)
				{
					frontValue+=ldg(rowValues15+rowPos15)*weight15;
					rowPos15++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, typename T>
static __device__ void DifMul_32(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B){
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread
	T* rowValues8;uint* rowIndices8;int rowLength8=0;//The row for the thread
	T* rowValues9;uint* rowIndices9;int rowLength9=0;//The row for the thread
	T* rowValues10;uint* rowIndices10;int rowLength10=0;//The row for the thread
	T* rowValues11;uint* rowIndices11;int rowLength11=0;//The row for the thread
	T* rowValues12;uint* rowIndices12;int rowLength12=0;//The row for the thread
	T* rowValues13;uint* rowIndices13;int rowLength13=0;//The row for the thread
	T* rowValues14;uint* rowIndices14;int rowLength14=0;//The row for the thread
	T* rowValues15;uint* rowIndices15;int rowLength15=0;//The row for the thread
	T* rowValues16;uint* rowIndices16;int rowLength16=0;//The row for the thread
	T* rowValues17;uint* rowIndices17;int rowLength17=0;//The row for the thread
	T* rowValues18;uint* rowIndices18;int rowLength18=0;//The row for the thread
	T* rowValues19;uint* rowIndices19;int rowLength19=0;//The row for the thread
	T* rowValues20;uint* rowIndices20;int rowLength20=0;//The row for the thread
	T* rowValues21;uint* rowIndices21;int rowLength21=0;//The row for the thread
	T* rowValues22;uint* rowIndices22;int rowLength22=0;//The row for the thread
	T* rowValues23;uint* rowIndices23;int rowLength23=0;//The row for the thread
	T* rowValues24;uint* rowIndices24;int rowLength24=0;//The row for the thread
	T* rowValues25;uint* rowIndices25;int rowLength25=0;//The row for the thread
	T* rowValues26;uint* rowIndices26;int rowLength26=0;//The row for the thread
	T* rowValues27;uint* rowIndices27;int rowLength27=0;//The row for the thread
	T* rowValues28;uint* rowIndices28;int rowLength28=0;//The row for the thread
	T* rowValues29;uint* rowIndices29;int rowLength29=0;//The row for the thread
	T* rowValues30;uint* rowIndices30;int rowLength30=0;//The row for the thread
	T* rowValues31;uint* rowIndices31;int rowLength31=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	T weight8=0;//The weight for the row
	T weight9=0;//The weight for the row
	T weight10=0;//The weight for the row
	T weight11=0;//The weight for the row
	T weight12=0;//The weight for the row
	T weight13=0;//The weight for the row
	T weight14=0;//The weight for the row
	T weight15=0;//The weight for the row
	T weight16=0;//The weight for the row
	T weight17=0;//The weight for the row
	T weight18=0;//The weight for the row
	T weight19=0;//The weight for the row
	T weight20=0;//The weight for the row
	T weight21=0;//The weight for the row
	T weight22=0;//The weight for the row
	T weight23=0;//The weight for the row
	T weight24=0;//The weight for the row
	T weight25=0;//The weight for the row
	T weight26=0;//The weight for the row
	T weight27=0;//The weight for the row
	T weight28=0;//The weight for the row
	T weight29=0;//The weight for the row
	T weight30=0;//The weight for the row
	T weight31=0;//The weight for the row
	int t=(threadIdx.x+1)*32;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint r25=ldg(a.Indices()+d0+25);
		uint r26=ldg(a.Indices()+d0+26);
		uint r27=ldg(a.Indices()+d0+27);
		uint r28=ldg(a.Indices()+d0+28);//uint rowIndex=a.Index(thread);		
		uint r29=ldg(a.Indices()+d0+29);
		uint r30=ldg(a.Indices()+d0+30);
		uint r31=ldg(a.Indices()+d0+31);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		uint rowStart25=ldg(B.RowStarts()+r25);
		uint rowStart26=ldg(B.RowStarts()+r26);
		uint rowStart27=ldg(B.RowStarts()+r27);
		uint rowStart28=ldg(B.RowStarts()+r28);
		uint rowStart29=ldg(B.RowStarts()+r29);
		uint rowStart30=ldg(B.RowStarts()+r30);
		uint rowStart31=ldg(B.RowStarts()+r31);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=ldg(B.RowStarts()+r25+1)-rowStart25;
		rowLength26=ldg(B.RowStarts()+r26+1)-rowStart26;
		rowLength27=ldg(B.RowStarts()+r27+1)-rowStart27;
		rowLength28=ldg(B.RowStarts()+r28+1)-rowStart28;
		rowLength29=ldg(B.RowStarts()+r29+1)-rowStart29;
		rowLength30=ldg(B.RowStarts()+r30+1)-rowStart30;
		rowLength31=ldg(B.RowStarts()+r31+1)-rowStart31;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		rowValues25=B.Values()+rowStart25;
		rowIndices25=B.ColIndices()+rowStart25;
		rowValues26=B.Values()+rowStart26;
		rowIndices26=B.ColIndices()+rowStart26;
		rowValues27=B.Values()+rowStart27;
		rowIndices27=B.ColIndices()+rowStart27;
		rowValues28=B.Values()+rowStart28;
		rowIndices28=B.ColIndices()+rowStart28;
		rowValues29=B.Values()+rowStart29;
		rowIndices29=B.ColIndices()+rowStart29;
		rowValues30=B.Values()+rowStart30;
		rowIndices30=B.ColIndices()+rowStart30;
		rowValues31=B.Values()+rowStart31;
		rowIndices31=B.ColIndices()+rowStart31;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
		weight25=ldg(a.Values()+d0+25);//a.Value(thread);
		weight26=ldg(a.Values()+d0+26);//a.Value(thread);
		weight27=ldg(a.Values()+d0+27);//a.Value(thread);
		weight28=ldg(a.Values()+d0+28);//a.Value(thread);
		weight29=ldg(a.Values()+d0+29);//a.Value(thread);
		weight30=ldg(a.Values()+d0+30);//a.Value(thread);
		weight31=ldg(a.Values()+d0+31);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%32==31
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint r25=ldg(a.Indices()+d0+25);
		uint r26=ldg(a.Indices()+d0+26);
		uint r27=ldg(a.Indices()+d0+27);
		uint r28=ldg(a.Indices()+d0+28);//uint rowIndex=a.Index(thread);		
		uint r29=ldg(a.Indices()+d0+29);
		uint r30=ldg(a.Indices()+d0+30);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		uint rowStart25=ldg(B.RowStarts()+r25);
		uint rowStart26=ldg(B.RowStarts()+r26);
		uint rowStart27=ldg(B.RowStarts()+r27);
		uint rowStart28=ldg(B.RowStarts()+r28);
		uint rowStart29=ldg(B.RowStarts()+r29);
		uint rowStart30=ldg(B.RowStarts()+r30);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=ldg(B.RowStarts()+r25+1)-rowStart25;
		rowLength26=ldg(B.RowStarts()+r26+1)-rowStart26;
		rowLength27=ldg(B.RowStarts()+r27+1)-rowStart27;
		rowLength28=ldg(B.RowStarts()+r28+1)-rowStart28;
		rowLength29=ldg(B.RowStarts()+r29+1)-rowStart29;
		rowLength30=ldg(B.RowStarts()+r30+1)-rowStart30;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		rowValues25=B.Values()+rowStart25;
		rowIndices25=B.ColIndices()+rowStart25;
		rowValues26=B.Values()+rowStart26;
		rowIndices26=B.ColIndices()+rowStart26;
		rowValues27=B.Values()+rowStart27;
		rowIndices27=B.ColIndices()+rowStart27;
		rowValues28=B.Values()+rowStart28;
		rowIndices28=B.ColIndices()+rowStart28;
		rowValues29=B.Values()+rowStart29;
		rowIndices29=B.ColIndices()+rowStart29;
		rowValues30=B.Values()+rowStart30;
		rowIndices30=B.ColIndices()+rowStart30;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
		weight25=ldg(a.Values()+d0+25);//a.Value(thread);
		weight26=ldg(a.Values()+d0+26);//a.Value(thread);
		weight27=ldg(a.Values()+d0+27);//a.Value(thread);
		weight28=ldg(a.Values()+d0+28);//a.Value(thread);
		weight29=ldg(a.Values()+d0+29);//a.Value(thread);
		weight30=ldg(a.Values()+d0+30);//a.Value(thread);
	}	
	else if(t-2==a.NonZeroCount())  //a.NonZeroCount()%32==30
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint r25=ldg(a.Indices()+d0+25);
		uint r26=ldg(a.Indices()+d0+26);
		uint r27=ldg(a.Indices()+d0+27);
		uint r28=ldg(a.Indices()+d0+28);//uint rowIndex=a.Index(thread);		
		uint r29=ldg(a.Indices()+d0+29);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		uint rowStart25=ldg(B.RowStarts()+r25);
		uint rowStart26=ldg(B.RowStarts()+r26);
		uint rowStart27=ldg(B.RowStarts()+r27);
		uint rowStart28=ldg(B.RowStarts()+r28);
		uint rowStart29=ldg(B.RowStarts()+r29);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=ldg(B.RowStarts()+r25+1)-rowStart25;
		rowLength26=ldg(B.RowStarts()+r26+1)-rowStart26;
		rowLength27=ldg(B.RowStarts()+r27+1)-rowStart27;
		rowLength28=ldg(B.RowStarts()+r28+1)-rowStart28;
		rowLength29=ldg(B.RowStarts()+r29+1)-rowStart29;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		rowValues25=B.Values()+rowStart25;
		rowIndices25=B.ColIndices()+rowStart25;
		rowValues26=B.Values()+rowStart26;
		rowIndices26=B.ColIndices()+rowStart26;
		rowValues27=B.Values()+rowStart27;
		rowIndices27=B.ColIndices()+rowStart27;
		rowValues28=B.Values()+rowStart28;
		rowIndices28=B.ColIndices()+rowStart28;
		rowValues29=B.Values()+rowStart29;
		rowIndices29=B.ColIndices()+rowStart29;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
		weight25=ldg(a.Values()+d0+25);//a.Value(thread);
		weight26=ldg(a.Values()+d0+26);//a.Value(thread);
		weight27=ldg(a.Values()+d0+27);//a.Value(thread);
		weight28=ldg(a.Values()+d0+28);//a.Value(thread);
		weight29=ldg(a.Values()+d0+29);//a.Value(thread);
	}	
	else if(t-3==a.NonZeroCount())  //a.NonZeroCount()%32==29
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint r25=ldg(a.Indices()+d0+25);
		uint r26=ldg(a.Indices()+d0+26);
		uint r27=ldg(a.Indices()+d0+27);
		uint r28=ldg(a.Indices()+d0+28);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		uint rowStart25=ldg(B.RowStarts()+r25);
		uint rowStart26=ldg(B.RowStarts()+r26);
		uint rowStart27=ldg(B.RowStarts()+r27);
		uint rowStart28=ldg(B.RowStarts()+r28);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=ldg(B.RowStarts()+r25+1)-rowStart25;
		rowLength26=ldg(B.RowStarts()+r26+1)-rowStart26;
		rowLength27=ldg(B.RowStarts()+r27+1)-rowStart27;
		rowLength28=ldg(B.RowStarts()+r28+1)-rowStart28;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		rowValues25=B.Values()+rowStart25;
		rowIndices25=B.ColIndices()+rowStart25;
		rowValues26=B.Values()+rowStart26;
		rowIndices26=B.ColIndices()+rowStart26;
		rowValues27=B.Values()+rowStart27;
		rowIndices27=B.ColIndices()+rowStart27;
		rowValues28=B.Values()+rowStart28;
		rowIndices28=B.ColIndices()+rowStart28;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
		weight25=ldg(a.Values()+d0+25);//a.Value(thread);
		weight26=ldg(a.Values()+d0+26);//a.Value(thread);
		weight27=ldg(a.Values()+d0+27);//a.Value(thread);
		weight28=ldg(a.Values()+d0+28);//a.Value(thread);
	}	
	else if(t-4==a.NonZeroCount())  //a.NonZeroCount()%32==28
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint r25=ldg(a.Indices()+d0+25);
		uint r26=ldg(a.Indices()+d0+26);
		uint r27=ldg(a.Indices()+d0+27);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		uint rowStart25=ldg(B.RowStarts()+r25);
		uint rowStart26=ldg(B.RowStarts()+r26);
		uint rowStart27=ldg(B.RowStarts()+r27);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=ldg(B.RowStarts()+r25+1)-rowStart25;
		rowLength26=ldg(B.RowStarts()+r26+1)-rowStart26;
		rowLength27=ldg(B.RowStarts()+r27+1)-rowStart27;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		rowValues25=B.Values()+rowStart25;
		rowIndices25=B.ColIndices()+rowStart25;
		rowValues26=B.Values()+rowStart26;
		rowIndices26=B.ColIndices()+rowStart26;
		rowValues27=B.Values()+rowStart27;
		rowIndices27=B.ColIndices()+rowStart27;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
		weight25=ldg(a.Values()+d0+25);//a.Value(thread);
		weight26=ldg(a.Values()+d0+26);//a.Value(thread);
		weight27=ldg(a.Values()+d0+27);//a.Value(thread);
	}	
	else if(t-5==a.NonZeroCount())  //a.NonZeroCount()%32==27
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint r25=ldg(a.Indices()+d0+25);
		uint r26=ldg(a.Indices()+d0+26);
		uint r27=ldg(a.Indices()+d0+27);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		uint rowStart25=ldg(B.RowStarts()+r25);
		uint rowStart26=ldg(B.RowStarts()+r26);
		uint rowStart27=ldg(B.RowStarts()+r27);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=ldg(B.RowStarts()+r25+1)-rowStart25;
		rowLength26=ldg(B.RowStarts()+r26+1)-rowStart26;
		rowLength27=ldg(B.RowStarts()+r27+1)-rowStart27;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		rowValues25=B.Values()+rowStart25;
		rowIndices25=B.ColIndices()+rowStart25;
		rowValues26=B.Values()+rowStart26;
		rowIndices26=B.ColIndices()+rowStart26;
		rowValues27=B.Values()+rowStart27;
		rowIndices27=B.ColIndices()+rowStart27;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
		weight25=ldg(a.Values()+d0+25);//a.Value(thread);
		weight26=ldg(a.Values()+d0+26);//a.Value(thread);
		weight27=ldg(a.Values()+d0+27);//a.Value(thread);
	}	
	else if(t-5==a.NonZeroCount())  //a.NonZeroCount()%32==27
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint r25=ldg(a.Indices()+d0+25);
		uint r26=ldg(a.Indices()+d0+26);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		uint rowStart25=ldg(B.RowStarts()+r25);
		uint rowStart26=ldg(B.RowStarts()+r26);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=ldg(B.RowStarts()+r25+1)-rowStart25;
		rowLength26=ldg(B.RowStarts()+r26+1)-rowStart26;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		rowValues25=B.Values()+rowStart25;
		rowIndices25=B.ColIndices()+rowStart25;
		rowValues26=B.Values()+rowStart26;
		rowIndices26=B.ColIndices()+rowStart26;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
		weight25=ldg(a.Values()+d0+25);//a.Value(thread);
		weight26=ldg(a.Values()+d0+26);//a.Value(thread);
	}	
	else if(t-6==a.NonZeroCount())  //a.NonZeroCount()%32==26
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint r25=ldg(a.Indices()+d0+25);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		uint rowStart25=ldg(B.RowStarts()+r25);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=ldg(B.RowStarts()+r25+1)-rowStart25;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		rowValues25=B.Values()+rowStart25;
		rowIndices25=B.ColIndices()+rowStart25;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
		weight25=ldg(a.Values()+d0+25);//a.Value(thread);
	}	
	else if(t-7==a.NonZeroCount())  //a.NonZeroCount()%32==25
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint r24=ldg(a.Indices()+d0+24);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		uint rowStart24=ldg(B.RowStarts()+r24);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=ldg(B.RowStarts()+r24+1)-rowStart24;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		rowValues24=B.Values()+rowStart24;
		rowIndices24=B.ColIndices()+rowStart24;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
		weight24=ldg(a.Values()+d0+24);//a.Value(thread);
	}	
	else if(t-8==a.NonZeroCount())  //a.NonZeroCount()%32==24
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint r23=ldg(a.Indices()+d0+23);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		uint rowStart23=ldg(B.RowStarts()+r23);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=ldg(B.RowStarts()+r23+1)-rowStart23;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		rowValues23=B.Values()+rowStart23;
		rowIndices23=B.ColIndices()+rowStart23;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
		weight23=ldg(a.Values()+d0+23);//a.Value(thread);
	}	
	else if(t-9==a.NonZeroCount())  //a.NonZeroCount()%32==23
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint r22=ldg(a.Indices()+d0+22);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		uint rowStart22=ldg(B.RowStarts()+r22);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=ldg(B.RowStarts()+r22+1)-rowStart22;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		rowValues22=B.Values()+rowStart22;
		rowIndices22=B.ColIndices()+rowStart22;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
		weight22=ldg(a.Values()+d0+22);//a.Value(thread);
	}	
	else if(t-10==a.NonZeroCount())  //a.NonZeroCount()%32==22
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint r21=ldg(a.Indices()+d0+21);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		uint rowStart21=ldg(B.RowStarts()+r21);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=ldg(B.RowStarts()+r21+1)-rowStart21;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		rowValues21=B.Values()+rowStart21;
		rowIndices21=B.ColIndices()+rowStart21;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
		weight21=ldg(a.Values()+d0+21);//a.Value(thread);
	}	
	else if(t-11==a.NonZeroCount())  //a.NonZeroCount()%32==21
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint r20=ldg(a.Indices()+d0+20);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		uint rowStart20=ldg(B.RowStarts()+r20);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=ldg(B.RowStarts()+r20+1)-rowStart20;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		rowValues20=B.Values()+rowStart20;
		rowIndices20=B.ColIndices()+rowStart20;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
		weight20=ldg(a.Values()+d0+20);//a.Value(thread);
	}	
	else if(t-12==a.NonZeroCount())  //a.NonZeroCount()%32==20
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint r19=ldg(a.Indices()+d0+19);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		uint rowStart19=ldg(B.RowStarts()+r19);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=ldg(B.RowStarts()+r19+1)-rowStart19;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		rowValues19=B.Values()+rowStart19;
		rowIndices19=B.ColIndices()+rowStart19;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
		weight19=ldg(a.Values()+d0+19);//a.Value(thread);
	}	
	else if(t-13==a.NonZeroCount())  //a.NonZeroCount()%32==19
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint r18=ldg(a.Indices()+d0+18);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		uint rowStart18=ldg(B.RowStarts()+r18);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=ldg(B.RowStarts()+r18+1)-rowStart18;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		rowValues18=B.Values()+rowStart18;
		rowIndices18=B.ColIndices()+rowStart18;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
		weight18=ldg(a.Values()+d0+18);//a.Value(thread);
	}	
	else if(t-14==a.NonZeroCount())  //a.NonZeroCount()%32==18
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint r17=ldg(a.Indices()+d0+17);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		uint rowStart17=ldg(B.RowStarts()+r17);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=ldg(B.RowStarts()+r17+1)-rowStart17;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		rowValues17=B.Values()+rowStart17;
		rowIndices17=B.ColIndices()+rowStart17;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
		weight17=ldg(a.Values()+d0+17);//a.Value(thread);
	}	
	else if(t-15==a.NonZeroCount())  //a.NonZeroCount()%32==17
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint r16=ldg(a.Indices()+d0+16);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		uint rowStart16=ldg(B.RowStarts()+r16);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=ldg(B.RowStarts()+r16+1)-rowStart16;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		rowValues16=B.Values()+rowStart16;
		rowIndices16=B.ColIndices()+rowStart16;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
		weight16=ldg(a.Values()+d0+16);//a.Value(thread);
	}
	else if(t-16==a.NonZeroCount())  //a.NonZeroCount()%32==16
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
	}
	else if(t-17==a.NonZeroCount())  //a.NonZeroCount()%32==15
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
	}
	else if(t-18==a.NonZeroCount())  //a.NonZeroCount()%32==14
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
	}
	else if(t-19==a.NonZeroCount())  //a.NonZeroCount()%32==13
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
	}
	else if(t-20==a.NonZeroCount())  //a.NonZeroCount()%32==12
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
	}
	else if(t-21==a.NonZeroCount())  //a.NonZeroCount()%32==11
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
	}	
	else if(t-22==a.NonZeroCount())  //a.NonZeroCount()%32==10
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
	}	
	else if(t-23==a.NonZeroCount())  //a.NonZeroCount()%32==9
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
	}	
	else if(t-24==a.NonZeroCount())  //a.NonZeroCount()%32==8
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
	}	
	else if(t-25==a.NonZeroCount())  //a.NonZeroCount()%32==7
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
	}	
	else if(t-26==a.NonZeroCount())  //a.NonZeroCount()%32==6
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
	}	
	else if(t-27==a.NonZeroCount())  //a.NonZeroCount()%32==5
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
	}	
	else if(t-28==a.NonZeroCount())  //a.NonZeroCount()%32==4
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}	
	else if(t-29==a.NonZeroCount())  //a.NonZeroCount()%32==3
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}	
	else if(t-30==a.NonZeroCount())  //a.NonZeroCount()%32==2
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}	
	else if(t-31==a.NonZeroCount())  //a.NonZeroCount()%32==1
	{
		uint d0=threadIdx.x*32;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}	
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowLength16=0;
		rowLength17=0;
		rowLength18=0;
		rowLength19=0;
		rowLength20=0;
		rowLength21=0;
		rowLength22=0;
		rowLength23=0;
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int rowPos8=0;//Current position into row
	int rowPos9=0;//Current position into row
	int rowPos10=0;//Current position into row
	int rowPos11=0;//Current position into row
	int rowPos12=0;//Current position into row
	int rowPos13=0;//Current position into row
	int rowPos14=0;//Current position into row
	int rowPos15=0;//Current position into row
	int rowPos16=0;//Current position into row
	int rowPos17=0;//Current position into row
	int rowPos18=0;//Current position into row
	int rowPos19=0;//Current position into row
	int rowPos20=0;//Current position into row
	int rowPos21=0;//Current position into row
	int rowPos22=0;//Current position into row
	int rowPos23=0;//Current position into row
	int rowPos24=0;//Current position into row
	int rowPos25=0;//Current position into row
	int rowPos26=0;//Current position into row
	int rowPos27=0;//Current position into row
	int rowPos28=0;//Current position into row
	int rowPos29=0;//Current position into row
	int rowPos30=0;//Current position into row
	int rowPos31=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	int index8=intMax;
	int index9=intMax;
	int index10=intMax;
	int index11=intMax;
	int index12=intMax;
	int index13=intMax;
	int index14=intMax;
	int index15=intMax;
	int index16=intMax;
	int index17=intMax;
	int index18=intMax;
	int index19=intMax;
	int index20=intMax;
	int index21=intMax;
	int index22=intMax;
	int index23=intMax;
	int index24=intMax;
	int index25=intMax;
	int index26=intMax;
	int index27=intMax;
	int index28=intMax;
	int index29=intMax;
	int index30=intMax;
	int index31=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}
	if(rowPos8<rowLength8){
		index8=ldg(rowIndices8+rowPos8);
	}
	if(rowPos9<rowLength9){
		index9=ldg(rowIndices9+rowPos9);
	}
	if(rowPos10<rowLength10){
		index10=ldg(rowIndices10+rowPos10);
	}
	if(rowPos11<rowLength11){
		index11=ldg(rowIndices11+rowPos11);
	}
	if(rowPos12<rowLength12){
		index12=ldg(rowIndices12+rowPos12);
	}
	if(rowPos13<rowLength13){
		index13=ldg(rowIndices13+rowPos13);
	}
	if(rowPos14<rowLength14){
		index14=ldg(rowIndices14+rowPos14);
	}
	if(rowPos15<rowLength15){
		index15=ldg(rowIndices15+rowPos15);
	}
	if(rowPos16<rowLength16){
		index16=ldg(rowIndices16+rowPos16);
	}
	if(rowPos17<rowLength17){
		index17=ldg(rowIndices17+rowPos17);
	}
	if(rowPos18<rowLength18){
		index18=ldg(rowIndices18+rowPos18);
	}
	if(rowPos19<rowLength19){
		index19=ldg(rowIndices19+rowPos19);
	}
	if(rowPos20<rowLength20){
		index20=ldg(rowIndices20+rowPos20);
	}
	if(rowPos21<rowLength21){
		index21=ldg(rowIndices21+rowPos21);
	}
	if(rowPos22<rowLength22){
		index22=ldg(rowIndices22+rowPos22);
	}
	if(rowPos23<rowLength23){
		index23=ldg(rowIndices23+rowPos23);
	}
	if(rowPos24<rowLength24){
		index24=ldg(rowIndices24+rowPos24);
	}
	if(rowPos25<rowLength25){
		index25=ldg(rowIndices25+rowPos25);
	}
	if(rowPos26<rowLength26){
		index26=ldg(rowIndices26+rowPos26);
	}
	if(rowPos27<rowLength27){
		index27=ldg(rowIndices27+rowPos27);
	}
	if(rowPos28<rowLength28){
		index28=ldg(rowIndices28+rowPos28);
	}
	if(rowPos29<rowLength29){
		index29=ldg(rowIndices29+rowPos29);
	}
	if(rowPos30<rowLength30){
		index30=ldg(rowIndices30+rowPos30);
	}
	if(rowPos31<rowLength31){
		index31=ldg(rowIndices31+rowPos31);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	min_index=index8<min_index?index8:min_index;
	min_index=index9<min_index?index9:min_index;
	min_index=index10<min_index?index10:min_index;
	min_index=index11<min_index?index11:min_index;
	min_index=index12<min_index?index12:min_index;
	min_index=index13<min_index?index13:min_index;
	min_index=index14<min_index?index14:min_index;
	min_index=index15<min_index?index15:min_index;
	min_index=index16<min_index?index16:min_index;
	min_index=index17<min_index?index17:min_index;
	min_index=index18<min_index?index18:min_index;
	min_index=index19<min_index?index19:min_index;
	min_index=index20<min_index?index20:min_index;
	min_index=index21<min_index?index21:min_index;
	min_index=index22<min_index?index22:min_index;
	min_index=index23<min_index?index23:min_index;
	min_index=index24<min_index?index24:min_index;
	min_index=index25<min_index?index25:min_index;
	min_index=index26<min_index?index26:min_index;
	min_index=index27<min_index?index27:min_index;
	min_index=index28<min_index?index28:min_index;
	min_index=index29<min_index?index29:min_index;
	min_index=index30<min_index?index30:min_index;
	min_index=index31<min_index?index31:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=ldg(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=ldg(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=ldg(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=ldg(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
		if(index8==min_index)
		{
			frontValue+=ldg(rowValues8+rowPos8)*weight8;
			rowPos8++;
		}
		if(index9==min_index)
		{
			frontValue+=ldg(rowValues9+rowPos9)*weight9;
			rowPos9++;
		}
		if(index10==min_index)
		{
			frontValue+=ldg(rowValues10+rowPos10)*weight10;
			rowPos10++;
		}
		if(index11==min_index)
		{
			frontValue+=ldg(rowValues11+rowPos11)*weight11;
			rowPos11++;
		}
		if(index12==min_index)
		{
			frontValue+=ldg(rowValues12+rowPos12)*weight12;
			rowPos12++;
		}
		if(index13==min_index)
		{
			frontValue+=ldg(rowValues13+rowPos13)*weight13;
			rowPos13++;
		}
		if(index14==min_index)
		{
			frontValue+=ldg(rowValues14+rowPos14)*weight14;
			rowPos14++;
		}
		if(index15==min_index)
		{
			frontValue+=ldg(rowValues15+rowPos15)*weight15;
			rowPos15++;
		}
		if(index16==min_index)
		{
			frontValue+=ldg(rowValues16+rowPos16)*weight16;
			rowPos16++;
		}
		if(index17==min_index)
		{
			frontValue+=ldg(rowValues17+rowPos17)*weight17;
			rowPos17++;
		}
		if(index18==min_index)
		{
			frontValue+=ldg(rowValues18+rowPos18)*weight18;
			rowPos18++;
		}
		if(index19==min_index)
		{
			frontValue+=ldg(rowValues19+rowPos19)*weight19;
			rowPos19++;
		}
		if(index20==min_index)
		{
			frontValue+=ldg(rowValues20+rowPos20)*weight20;
			rowPos20++;
		}
		if(index21==min_index)
		{
			frontValue+=ldg(rowValues21+rowPos21)*weight21;
			rowPos21++;
		}
		if(index22==min_index)
		{
			frontValue+=ldg(rowValues22+rowPos22)*weight22;
			rowPos22++;
		}
		if(index23==min_index)
		{
			frontValue+=ldg(rowValues23+rowPos23)*weight23;
			rowPos23++;
		}
		if(index24==min_index)
		{
			frontValue+=ldg(rowValues24+rowPos24)*weight24;
			rowPos24++;
		}
		if(index25==min_index)
		{
			frontValue+=ldg(rowValues25+rowPos25)*weight25;
			rowPos25++;
		}
		if(index26==min_index)
		{
			frontValue+=ldg(rowValues26+rowPos26)*weight26;
			rowPos26++;
		}
		if(index27==min_index)
		{
			frontValue+=ldg(rowValues27+rowPos27)*weight27;
			rowPos27++;
		}
		if(index28==min_index)
		{
			frontValue+=ldg(rowValues28+rowPos28)*weight28;
			rowPos28++;
		}
		if(index29==min_index)
		{
			frontValue+=ldg(rowValues29+rowPos29)*weight29;
			rowPos29++;
		}
		if(index30==min_index)
		{
			frontValue+=ldg(rowValues30+rowPos30)*weight30;
			rowPos30++;
		}
		if(index31==min_index)
		{
			frontValue+=ldg(rowValues31+rowPos31)*weight31;
			rowPos31++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
	//		frontIndex=index0>index1?index1:index0;
	//		frontValue=index0>index1?ldg(rowValues1+rowPos1)*weight1:ldg(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//	if(threadIdx.x==1&&threadIdx.y==0)
	//	{
	//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
	//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
	//		printf("weight0=%f,weight1=%f,weight2=%f,weight3=%f,weight4=%f,weight5=%f,weight6=%f,weight7=%f\n",weight0,weight1,weight2,weight3,weight4,weight5,weight6,weight7);
	//		printf("weight8=%f,weight9=%f,weight10=%f,weight11=%f,weight12=%f,weight13=%f,weight14=%f,weight15=%f\n",weight8,weight9,weight10,weight11,weight12,weight13,weight14,weight15);
	//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
	//		printf("minFront=%d\n",minFront);
	//		printf("------------------------------------\n");
	//	}
	//	if(threadIdx.x==0&&threadIdx.y==0)
	//	{
	//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
	//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
	//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
	//		printf("minFront=%d\n",minFront);
	//		printf("------------------------------------\n");
	//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}
			if(rowPos8<rowLength8){
				index8=ldg(rowIndices8+rowPos8);
			}
			else{
				index8=intMax;
			}
			if(rowPos9<rowLength9){
				index9=ldg(rowIndices9+rowPos9);
			}
			else{
				index9=intMax;
			}
			if(rowPos10<rowLength10){
				index10=ldg(rowIndices10+rowPos10);
			}
			else{
				index10=intMax;
			}
			if(rowPos11<rowLength11){
				index11=ldg(rowIndices11+rowPos11);
			}
			else{
				index11=intMax;
			}
			if(rowPos12<rowLength12){
				index12=ldg(rowIndices12+rowPos12);
			}
			else{
				index12=intMax;
			}
			if(rowPos13<rowLength13){
				index13=ldg(rowIndices13+rowPos13);
			}
			else{
				index13=intMax;
			}
			if(rowPos14<rowLength14){
				index14=ldg(rowIndices14+rowPos14);
			}
			else{
				index14=intMax;
			}
			if(rowPos15<rowLength15){
				index15=ldg(rowIndices15+rowPos15);
			}
			else{
				index15=intMax;
			}
			if(rowPos16<rowLength16){
				index16=ldg(rowIndices16+rowPos16);
			}
			else{
				index16=intMax;
			}
			if(rowPos17<rowLength17){
				index17=ldg(rowIndices17+rowPos17);
			}
			else{
				index17=intMax;
			}
			if(rowPos18<rowLength18){
				index18=ldg(rowIndices18+rowPos18);
			}
			else{
				index18=intMax;
			}
			if(rowPos19<rowLength19){
				index19=ldg(rowIndices19+rowPos19);
			}
			else{
				index19=intMax;
			}
			if(rowPos20<rowLength20){
				index20=ldg(rowIndices20+rowPos20);
			}
			else{
				index20=intMax;
			}
			if(rowPos21<rowLength21){
				index21=ldg(rowIndices21+rowPos21);
			}
			else{
				index21=intMax;
			}
			if(rowPos22<rowLength22){
				index22=ldg(rowIndices22+rowPos22);
			}
			else{
				index22=intMax;
			}
			if(rowPos23<rowLength23){
				index23=ldg(rowIndices23+rowPos23);
			}
			else{
				index23=intMax;
			}
			if(rowPos24<rowLength24){
				index24=ldg(rowIndices24+rowPos24);
			}
			else{
				index24=intMax;
			}
			if(rowPos25<rowLength25){
				index25=ldg(rowIndices25+rowPos25);
			}
			else{
				index25=intMax;
			}
			if(rowPos26<rowLength26){
				index26=ldg(rowIndices26+rowPos26);
			}
			else{
				index26=intMax;
			}
			if(rowPos27<rowLength27){
				index27=ldg(rowIndices27+rowPos27);
			}
			else{
				index27=intMax;
			}
			if(rowPos28<rowLength28){
				index28=ldg(rowIndices28+rowPos28);
			}
			else{
				index28=intMax;
			}
			if(rowPos29<rowLength29){
				index29=ldg(rowIndices29+rowPos29);
			}
			else{
				index29=intMax;
			}
			if(rowPos30<rowLength30){
				index30=ldg(rowIndices30+rowPos30);
			}
			else{
				index30=intMax;
			}
			if(rowPos31<rowLength31){
				index31=ldg(rowIndices31+rowPos31);
			}
			else{
				index31=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			min_index=index8<min_index?index8:min_index;
			min_index=index9<min_index?index9:min_index;
			min_index=index10<min_index?index10:min_index;
			min_index=index11<min_index?index11:min_index;
			min_index=index12<min_index?index12:min_index;
			min_index=index13<min_index?index13:min_index;
			min_index=index14<min_index?index14:min_index;
			min_index=index15<min_index?index15:min_index;
			min_index=index16<min_index?index16:min_index;
			min_index=index17<min_index?index17:min_index;
			min_index=index18<min_index?index18:min_index;
			min_index=index19<min_index?index19:min_index;
			min_index=index20<min_index?index20:min_index;
			min_index=index21<min_index?index21:min_index;
			min_index=index22<min_index?index22:min_index;
			min_index=index23<min_index?index23:min_index;
			min_index=index24<min_index?index24:min_index;
			min_index=index25<min_index?index25:min_index;
			min_index=index26<min_index?index26:min_index;
			min_index=index27<min_index?index27:min_index;
			min_index=index28<min_index?index28:min_index;
			min_index=index29<min_index?index29:min_index;
			min_index=index30<min_index?index30:min_index;
			min_index=index31<min_index?index31:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=ldg(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=ldg(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=ldg(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=ldg(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
				if(index8==min_index)
				{
					frontValue+=ldg(rowValues8+rowPos8)*weight8;
					rowPos8++;
				}
				if(index9==min_index)
				{
					frontValue+=ldg(rowValues9+rowPos9)*weight9;
					rowPos9++;
				}
				if(index10==min_index)
				{
					frontValue+=ldg(rowValues10+rowPos10)*weight10;
					rowPos10++;
				}
				if(index11==min_index)
				{
					frontValue+=ldg(rowValues11+rowPos11)*weight11;
					rowPos11++;
				}
				if(index12==min_index)
				{
					frontValue+=ldg(rowValues12+rowPos12)*weight12;
					rowPos12++;
				}
				if(index13==min_index)
				{
					frontValue+=ldg(rowValues13+rowPos13)*weight13;
					rowPos13++;
				}
				if(index14==min_index)
				{
					frontValue+=ldg(rowValues14+rowPos14)*weight14;
					rowPos14++;
				}
				if(index15==min_index)
				{
					frontValue+=ldg(rowValues15+rowPos15)*weight15;
					rowPos15++;
				}
				if(index16==min_index)
				{
					frontValue+=ldg(rowValues16+rowPos16)*weight16;
					rowPos16++;
				}
				if(index17==min_index)
				{
					frontValue+=ldg(rowValues17+rowPos17)*weight17;
					rowPos17++;
				}
				if(index18==min_index)
				{
					frontValue+=ldg(rowValues18+rowPos18)*weight18;
					rowPos18++;
				}
				if(index19==min_index)
				{
					frontValue+=ldg(rowValues19+rowPos19)*weight19;
					rowPos19++;
				}
				if(index20==min_index)
				{
					frontValue+=ldg(rowValues20+rowPos20)*weight20;
					rowPos20++;
				}
				if(index21==min_index)
				{
					frontValue+=ldg(rowValues21+rowPos21)*weight21;
					rowPos21++;
				}
				if(index22==min_index)
				{
					frontValue+=ldg(rowValues22+rowPos22)*weight22;
					rowPos22++;
				}
				if(index23==min_index)
				{
					frontValue+=ldg(rowValues23+rowPos23)*weight23;
					rowPos23++;
				}
				if(index24==min_index)
				{
					frontValue+=ldg(rowValues24+rowPos24)*weight24;
					rowPos24++;
				}
				if(index25==min_index)
				{
					frontValue+=ldg(rowValues25+rowPos25)*weight25;
					rowPos25++;
				}
				if(index26==min_index)
				{
					frontValue+=ldg(rowValues26+rowPos26)*weight26;
					rowPos26++;
				}
				if(index27==min_index)
				{
					frontValue+=ldg(rowValues27+rowPos27)*weight27;
					rowPos27++;
				}
				if(index28==min_index)
				{
					frontValue+=ldg(rowValues28+rowPos28)*weight28;
					rowPos28++;
				}
				if(index29==min_index)
				{
					frontValue+=ldg(rowValues29+rowPos29)*weight29;
					rowPos29++;
				}
				if(index30==min_index)
				{
					frontValue+=ldg(rowValues30+rowPos30)*weight30;
					rowPos30++;
				}
				if(index31==min_index)
				{
					frontValue+=ldg(rowValues31+rowPos31)*weight31;
					rowPos31++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_1(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, T *c_val, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);

		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues;uint* rowIndices;int rowLength=0;//The row for the thread
	T weight=0;//The weight for the row
	if(threadIdx.x<a.NonZeroCount()){
		uint r=ldg(a.Indices()+threadIdx.x);//uint rowIndex=a.Index(thread);		
		uint rowStart=ldg(B.RowStarts()+r);
		rowLength=ldg(B.RowStarts()+r+1)-rowStart;
		rowValues=B.Values()+rowStart;
		rowIndices=B.ColIndices()+rowStart;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight=ldg(a.Values()+threadIdx.x);//a.Value(thread);
	}

	int rowPos=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread
	if(rowPos<rowLength){//Load the front index and row
		frontIndex=ldg(rowIndices+rowPos);//ldg: explicit cache usage
		frontValue=ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos<rowLength){
				frontValue=ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
				frontIndex=(int)ldg(rowIndices+rowPos);//ldg: explicit cache usage
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}

		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();

		sum=WarpSum<WarpSize>(sum);
		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_2(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, T *c_val, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	int t=(threadIdx.x+1)*2;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*2;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount()){

		uint d0=threadIdx.x*2;
		uint r0=ldg(a.Indices()+d0);
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowLength1=0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}

	if(index0<index1)
	{
		frontIndex=index0;
		frontValue=ldg(rowValues0+rowPos0)*weight0;
		rowPos0++;
	}
	else if(index0>index1)
	{
		frontIndex=index1;
		frontValue=ldg(rowValues1+rowPos1)*weight1;
		rowPos1++;
	}
	else
	{
		if(index0!=intMax)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0+ldg(rowValues1+rowPos1)*weight1;
			rowPos0++;
			rowPos1++;
		}
		else
		{
		}
	}


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(index0<index1)
			{
				frontIndex=index0;
				frontValue=ldg(rowValues0+rowPos0)*weight0;
				rowPos0++;
			}
			else if(index0>index1)
			{
				frontIndex=index1;
				frontValue=ldg(rowValues1+rowPos1)*weight1;
				rowPos1++;
			}
			else 
			{
				if(index0!=intMax)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0+ldg(rowValues1+rowPos1)*weight1;
					rowPos0++;
					rowPos1++;
				}
				else
				{
					frontIndex=intMax;
				}
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}

		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();

		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);
		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_4(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, T *c_val, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	int t=(threadIdx.x+1)*4;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*4;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%4==3
	{
		uint d0=threadIdx.x*4;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}
	else if(t-2==a.NonZeroCount()) //a.NonZeroCount()%4==2
	{
		uint d0=threadIdx.x*4;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-3==a.NonZeroCount()) //a.NonZeroCount()%4==1
	{
		uint d0=threadIdx.x*4;
		uint r0=ldg(a.Indices()+d0);
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);

	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
	}
	else
	{
		frontIndex=intMax;
	}


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}

		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();

		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);
		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_8(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, T *c_val, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	int t=(threadIdx.x+1)*8;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%8==7
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
	}
	else if(t-2==a.NonZeroCount()) //a.NonZeroCount()%8==6
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint r5=ldg(a.Indices()+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
	}
	else if(t-3==a.NonZeroCount())// a.NonZeroCount()%8==5
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
	}
	else if(t-4==a.NonZeroCount())// a.NonZeroCount()%8==4
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}
	else if(t-5==a.NonZeroCount())// a.NonZeroCount()%8==3
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}
	else if(t-6==a.NonZeroCount())// a.NonZeroCount()%8==2
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-7==a.NonZeroCount())// a.NonZeroCount()%8==1
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	frontIndex=min_index;

	
	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=ldg(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=ldg(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=ldg(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=ldg(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
//		frontIndex=index0>index1?index1:index0;
//		frontValue=index0>index1?ldg(rowValues1+rowPos1)*weight1:ldg(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

//	if(threadIdx.x==1&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
//	if(threadIdx.x==0&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=ldg(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=ldg(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=ldg(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=ldg(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}
		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();
		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		

		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		

	}
}
template<int WarpSize, int SegmentSize, int halfNUM, typename T>
static __device__ void MulOverWarp_8_halfdown(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, T *c_val, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(a.NonZeroCount()-halfNUM==0)//nothing to do
		return;
	else if(a.NonZeroCount()-halfNUM==1){//simply scale the vector (faster)
		T weight=a.Value(0+halfNUM);
		CSparseVector<T> b=B.GetRow(a.Index(0+halfNUM));
        for(int j=threadIdx.x; j<b.NonZeroCount(); j+=blockDim.x)
        {
            for(int i=0;i<dst.NonZeroCount();i++)
            {
                if(dst.Index(i)==b.Index(j))
                {
                    dst.Value(i)+=weight*b.Value(j);
                    break;
                }
            }
        
        }
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	int t=(threadIdx.x+1)*8;

	if(t<=a.NonZeroCount()-halfNUM){
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+halfNUM+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+halfNUM+d0+1);
		uint r2=ldg(a.Indices()+halfNUM+d0+2);
		uint r3=ldg(a.Indices()+halfNUM+d0+3);
		uint r4=ldg(a.Indices()+halfNUM+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+halfNUM+d0+5);
		uint r6=ldg(a.Indices()+halfNUM+d0+6);
		uint r7=ldg(a.Indices()+halfNUM+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+halfNUM+d0);//a.Value(thread);
		weight1=ldg(a.Values()+halfNUM+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+halfNUM+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+halfNUM+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+halfNUM+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+halfNUM+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+halfNUM+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+halfNUM+d0+7);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount()-halfNUM)  //a.NonZeroCount()%8==7
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+halfNUM+d0);
		uint r1=ldg(a.Indices()+halfNUM+d0+1);
		uint r2=ldg(a.Indices()+halfNUM+d0+2);
		uint r3=ldg(a.Indices()+halfNUM+d0+3);
		uint r4=ldg(a.Indices()+halfNUM+d0+4);
		uint r5=ldg(a.Indices()+halfNUM+d0+5);
		uint r6=ldg(a.Indices()+halfNUM+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+halfNUM+d0);//a.Value(thread);
		weight1=ldg(a.Values()+halfNUM+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+halfNUM+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+halfNUM+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+halfNUM+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+halfNUM+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+halfNUM+d0+6);//a.Value(thread);
	}
	else if(t-2==a.NonZeroCount()-halfNUM) //a.NonZeroCount()%8==6
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+halfNUM+d0);
		uint r1=ldg(a.Indices()+halfNUM+d0+1);
		uint r2=ldg(a.Indices()+halfNUM+d0+2);
		uint r3=ldg(a.Indices()+halfNUM+d0+3);
		uint r4=ldg(a.Indices()+halfNUM+d0+4);
		uint r5=ldg(a.Indices()+halfNUM+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+halfNUM+d0);//a.Value(thread);
		weight1=ldg(a.Values()+halfNUM+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+halfNUM+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+halfNUM+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+halfNUM+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+halfNUM+d0+5);//a.Value(thread);
	}
	else if(t-3==a.NonZeroCount()-halfNUM)// a.NonZeroCount()%8==5
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+halfNUM+d0);
		uint r1=ldg(a.Indices()+halfNUM+d0+1);
		uint r2=ldg(a.Indices()+halfNUM+d0+2);
		uint r3=ldg(a.Indices()+halfNUM+d0+3);
		uint r4=ldg(a.Indices()+halfNUM+d0+4);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+halfNUM+d0);//a.Value(thread);
		weight1=ldg(a.Values()+halfNUM+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+halfNUM+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+halfNUM+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+halfNUM+d0+4);//a.Value(thread);
	}
	else if(t-4==a.NonZeroCount()-halfNUM)// a.NonZeroCount()%8==4
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+halfNUM+d0);
		uint r1=ldg(a.Indices()+halfNUM+d0+1);
		uint r2=ldg(a.Indices()+halfNUM+d0+2);
		uint r3=ldg(a.Indices()+halfNUM+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+halfNUM+d0);//a.Value(thread);
		weight1=ldg(a.Values()+halfNUM+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+halfNUM+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+halfNUM+d0+3);//a.Value(thread);
	}
	else if(t-5==a.NonZeroCount()-halfNUM)// a.NonZeroCount()%8==3
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+halfNUM+d0);
		uint r1=ldg(a.Indices()+halfNUM+d0+1);
		uint r2=ldg(a.Indices()+halfNUM+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+halfNUM+d0);//a.Value(thread);
		weight1=ldg(a.Values()+halfNUM+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+halfNUM+d0+2);//a.Value(thread);
	}
	else if(t-6==a.NonZeroCount()-halfNUM)// a.NonZeroCount()%8==2
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+halfNUM+d0);
		uint r1=ldg(a.Indices()+halfNUM+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+halfNUM+d0);//a.Value(thread);
		weight1=ldg(a.Values()+halfNUM+d0+1);//a.Value(thread);
	}
	else if(t-7==a.NonZeroCount()-halfNUM)// a.NonZeroCount()%8==1
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+halfNUM+d0);
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+halfNUM+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	frontIndex=min_index;

	
	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=ldg(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=ldg(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=ldg(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=ldg(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
//		frontIndex=index0>index1?index1:index0;
//		frontValue=index0>index1?ldg(rowValues1+rowPos1)*weight1:ldg(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

//	int dstPos=0;

//	if(threadIdx.x==1&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
//	if(threadIdx.x==0&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=ldg(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=ldg(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=ldg(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=ldg(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}
		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();
		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		

		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)

            for(int i=0;i<dst.NonZeroCount();i++)
            {
                if(dst.Indices()[i]==bufferedIndex)
                {
                    dst.Values()[i]+=bufferedValue;
                    break;
                }
            }

//            dstPos+=blockDim.x;
			bufferPos=0;
		}		

	}
}

template<int WarpSize, int SegmentSize, int halfNUM, typename T>
static __device__ void MulOverWarp_8_halfup(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, T *c_val, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	int t=(threadIdx.x+1)*8;

	if(t<=halfNUM){
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
	}
	else if(t-1==halfNUM)  //a.NonZeroCount()%8==7
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
	}
	else if(t-2==halfNUM) //a.NonZeroCount()%8==6
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint r5=ldg(a.Indices()+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
	}
	else if(t-3==halfNUM)// a.NonZeroCount()%8==5
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
	}
	else if(t-4==halfNUM)// a.NonZeroCount()%8==4
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}
	else if(t-5==halfNUM)// a.NonZeroCount()%8==3
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}
	else if(t-6==halfNUM)// a.NonZeroCount()%8==2
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-7==halfNUM)// a.NonZeroCount()%8==1
	{
		uint d0=threadIdx.x*8;
		uint r0=ldg(a.Indices()+d0);
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	frontIndex=min_index;

	
	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=ldg(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=ldg(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=ldg(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=ldg(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
//		frontIndex=index0>index1?index1:index0;
//		frontValue=index0>index1?ldg(rowValues1+rowPos1)*weight1:ldg(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);


//	if(threadIdx.x==1&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
//	if(threadIdx.x==0&&threadIdx.y==0)
//	{
//		printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//		printf("index0=%d,index1=%d,index2=%d,index3=%d,index4=%d,index5=%d,index6=%d,index7=%d\n",index0,index1,index2,index3,index4,index5,index6,index7);
//		printf("frontIndex=%d,frontValue=%f\n",frontIndex,frontValue);
//		printf("minFront=%d\n",minFront);
//		printf("------------------------------------\n");
//	}
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=ldg(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=ldg(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=ldg(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=ldg(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}
		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();
		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		

		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
            for(int i=0; i<dst.NonZeroCount(); i++)
            {
                if(dst.Indices()[i] == bufferedIndex)
                {
                    dst.Values()[i] = bufferedValue;
                }
            }
//			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
//			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
//			dstPos+=blockDim.x;
			bufferPos=0;
		}		

	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarp_16(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, T *c_val, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread
	T* rowValues8;uint* rowIndices8;int rowLength8=0;//The row for the thread
	T* rowValues9;uint* rowIndices9;int rowLength9=0;//The row for the thread
	T* rowValues10;uint* rowIndices10;int rowLength10=0;//The row for the thread
	T* rowValues11;uint* rowIndices11;int rowLength11=0;//The row for the thread
	T* rowValues12;uint* rowIndices12;int rowLength12=0;//The row for the thread
	T* rowValues13;uint* rowIndices13;int rowLength13=0;//The row for the thread
	T* rowValues14;uint* rowIndices14;int rowLength14=0;//The row for the thread
	T* rowValues15;uint* rowIndices15;int rowLength15=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	T weight8=0;//The weight for the row
	T weight9=0;//The weight for the row
	T weight10=0;//The weight for the row
	T weight11=0;//The weight for the row
	T weight12=0;//The weight for the row
	T weight13=0;//The weight for the row
	T weight14=0;//The weight for the row
	T weight15=0;//The weight for the row
	int t=(threadIdx.x+1)*16;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%16==15
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
	}
	else if(t-2==a.NonZeroCount()) //a.NonZeroCount()%16==14
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
	}
	else if(t-3==a.NonZeroCount())// a.NonZeroCount()%16==13
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
	}
	else if(t-4==a.NonZeroCount())// a.NonZeroCount()%16==12
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
	}
	else if(t-5==a.NonZeroCount())// a.NonZeroCount()%16==11
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
	}
	else if(t-6==a.NonZeroCount())// a.NonZeroCount()%16==10
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
	}
	else if(t-7==a.NonZeroCount())// a.NonZeroCount()%16==9
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
	}
	else if(t-8==a.NonZeroCount())// a.NonZeroCount()%16==8
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
	}
	else if(t-9==a.NonZeroCount())// a.NonZeroCount()%16==7
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
	}
	else if(t-10==a.NonZeroCount())// a.NonZeroCount()%16==6
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
	}
	else if(t-11==a.NonZeroCount())// a.NonZeroCount()%16==5
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
	}
	else if(t-12==a.NonZeroCount())// a.NonZeroCount()%16==4
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}
	else if(t-13==a.NonZeroCount())// a.NonZeroCount()%16==3
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}
	else if(t-14==a.NonZeroCount())// a.NonZeroCount()%16==2
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-15==a.NonZeroCount())// a.NonZeroCount()%16==1
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int rowPos8=0;//Current position into row
	int rowPos9=0;//Current position into row
	int rowPos10=0;//Current position into row
	int rowPos11=0;//Current position into row
	int rowPos12=0;//Current position into row
	int rowPos13=0;//Current position into row
	int rowPos14=0;//Current position into row
	int rowPos15=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	int index8=intMax;
	int index9=intMax;
	int index10=intMax;
	int index11=intMax;
	int index12=intMax;
	int index13=intMax;
	int index14=intMax;
	int index15=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}
	if(rowPos8<rowLength8){
		index8=ldg(rowIndices8+rowPos8);
	}
	if(rowPos9<rowLength9){
		index9=ldg(rowIndices9+rowPos9);
	}
	if(rowPos10<rowLength10){
		index10=ldg(rowIndices10+rowPos10);
	}
	if(rowPos11<rowLength11){
		index11=ldg(rowIndices11+rowPos11);
	}
	if(rowPos12<rowLength12){
		index12=ldg(rowIndices12+rowPos12);
	}
	if(rowPos13<rowLength13){
		index13=ldg(rowIndices13+rowPos13);
	}
	if(rowPos14<rowLength14){
		index14=ldg(rowIndices14+rowPos14);
	}
	if(rowPos15<rowLength15){
		index15=ldg(rowIndices15+rowPos15);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	min_index=index8<min_index?index8:min_index;
	min_index=index9<min_index?index9:min_index;
	min_index=index10<min_index?index10:min_index;
	min_index=index11<min_index?index11:min_index;
	min_index=index12<min_index?index12:min_index;
	min_index=index13<min_index?index13:min_index;
	min_index=index14<min_index?index14:min_index;
	min_index=index15<min_index?index15:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=ldg(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=ldg(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=ldg(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=ldg(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
		if(index8==min_index)
		{
			frontValue+=ldg(rowValues8+rowPos8)*weight8;
			rowPos8++;
		}
		if(index9==min_index)
		{
			frontValue+=ldg(rowValues9+rowPos9)*weight9;
			rowPos9++;
		}
		if(index10==min_index)
		{
			frontValue+=ldg(rowValues10+rowPos10)*weight10;
			rowPos10++;
		}
		if(index11==min_index)
		{
			frontValue+=ldg(rowValues11+rowPos11)*weight11;
			rowPos11++;
		}
		if(index12==min_index)
		{
			frontValue+=ldg(rowValues12+rowPos12)*weight12;
			rowPos12++;
		}
		if(index13==min_index)
		{
			frontValue+=ldg(rowValues13+rowPos13)*weight13;
			rowPos13++;
		}
		if(index14==min_index)
		{
			frontValue+=ldg(rowValues14+rowPos14)*weight14;
			rowPos14++;
		}
		if(index15==min_index)
		{
			frontValue+=ldg(rowValues15+rowPos15)*weight15;
			rowPos15++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
	//		frontIndex=index0>index1?index1:index0;
	//		frontValue=index0>index1?ldg(rowValues1+rowPos1)*weight1:ldg(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}
			if(rowPos8<rowLength8){
				index8=ldg(rowIndices8+rowPos8);
			}
			else{
				index8=intMax;
			}
			if(rowPos9<rowLength9){
				index9=ldg(rowIndices9+rowPos9);
			}
			else{
				index9=intMax;
			}
			if(rowPos10<rowLength10){
				index10=ldg(rowIndices10+rowPos10);
			}
			else{
				index10=intMax;
			}
			if(rowPos11<rowLength11){
				index11=ldg(rowIndices11+rowPos11);
			}
			else{
				index11=intMax;
			}
			if(rowPos12<rowLength12){
				index12=ldg(rowIndices12+rowPos12);
			}
			else{
				index12=intMax;
			}
			if(rowPos13<rowLength13){
				index13=ldg(rowIndices13+rowPos13);
			}
			else{
				index13=intMax;
			}
			if(rowPos14<rowLength14){
				index14=ldg(rowIndices14+rowPos14);
			}
			else{
				index14=intMax;
			}
			if(rowPos15<rowLength15){
				index15=ldg(rowIndices15+rowPos15);
			}
			else{
				index15=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			min_index=index8<min_index?index8:min_index;
			min_index=index9<min_index?index9:min_index;
			min_index=index10<min_index?index10:min_index;
			min_index=index11<min_index?index11:min_index;
			min_index=index12<min_index?index12:min_index;
			min_index=index13<min_index?index13:min_index;
			min_index=index14<min_index?index14:min_index;
			min_index=index15<min_index?index15:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=ldg(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=ldg(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=ldg(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=ldg(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
				if(index8==min_index)
				{
					frontValue+=ldg(rowValues8+rowPos8)*weight8;
					rowPos8++;
				}
				if(index9==min_index)
				{
					frontValue+=ldg(rowValues9+rowPos9)*weight9;
					rowPos9++;
				}
				if(index10==min_index)
				{
					frontValue+=ldg(rowValues10+rowPos10)*weight10;
					rowPos10++;
				}
				if(index11==min_index)
				{
					frontValue+=ldg(rowValues11+rowPos11)*weight11;
					rowPos11++;
				}
				if(index12==min_index)
				{
					frontValue+=ldg(rowValues12+rowPos12)*weight12;
					rowPos12++;
				}
				if(index13==min_index)
				{
					frontValue+=ldg(rowValues13+rowPos13)*weight13;
					rowPos13++;
				}
				if(index14==min_index)
				{
					frontValue+=ldg(rowValues14+rowPos14)*weight14;
					rowPos14++;
				}
				if(index15==min_index)
				{
					frontValue+=ldg(rowValues15+rowPos15)*weight15;
					rowPos15++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}

		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();

		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}

template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarpColumn_16(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=b.Index(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
    uint* rowIndices0;int rowLength0=0;//The row for the thread
	uint* rowIndices1;int rowLength1=0;//The row for the thread
	uint* rowIndices2;int rowLength2=0;//The row for the thread
	uint* rowIndices3;int rowLength3=0;//The row for the thread
	uint* rowIndices4;int rowLength4=0;//The row for the thread
	uint* rowIndices5;int rowLength5=0;//The row for the thread
	uint* rowIndices6;int rowLength6=0;//The row for the thread
	uint* rowIndices7;int rowLength7=0;//The row for the thread
	uint* rowIndices8;int rowLength8=0;//The row for the thread
	uint* rowIndices9;int rowLength9=0;//The row for the thread
	uint* rowIndices10;int rowLength10=0;//The row for the thread
	uint* rowIndices11;int rowLength11=0;//The row for the thread
	uint* rowIndices12;int rowLength12=0;//The row for the thread
	uint* rowIndices13;int rowLength13=0;//The row for the thread
	uint* rowIndices14;int rowLength14=0;//The row for the thread
	uint* rowIndices15;int rowLength15=0;//The row for the thread
	int t=(threadIdx.x+1)*16;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		rowIndices8=B.ColIndices()+rowStart8;
		rowIndices9=B.ColIndices()+rowStart9;
		rowIndices10=B.ColIndices()+rowStart10;
		rowIndices11=B.ColIndices()+rowStart11;
		rowIndices12=B.ColIndices()+rowStart12;
		rowIndices13=B.ColIndices()+rowStart13;
		rowIndices14=B.ColIndices()+rowStart14;
		rowIndices15=B.ColIndices()+rowStart15;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%16==15
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		rowIndices8=B.ColIndices()+rowStart8;
		rowIndices9=B.ColIndices()+rowStart9;
		rowIndices10=B.ColIndices()+rowStart10;
		rowIndices11=B.ColIndices()+rowStart11;
		rowIndices12=B.ColIndices()+rowStart12;
		rowIndices13=B.ColIndices()+rowStart13;
		rowIndices14=B.ColIndices()+rowStart14;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-2==a.NonZeroCount()) //a.NonZeroCount()%16==14
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		rowIndices8=B.ColIndices()+rowStart8;
		rowIndices9=B.ColIndices()+rowStart9;
		rowIndices10=B.ColIndices()+rowStart10;
		rowIndices11=B.ColIndices()+rowStart11;
		rowIndices12=B.ColIndices()+rowStart12;
		rowIndices13=B.ColIndices()+rowStart13;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-3==a.NonZeroCount())// a.NonZeroCount()%16==13
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		rowIndices8=B.ColIndices()+rowStart8;
		rowIndices9=B.ColIndices()+rowStart9;
		rowIndices10=B.ColIndices()+rowStart10;
		rowIndices11=B.ColIndices()+rowStart11;
		rowIndices12=B.ColIndices()+rowStart12;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-4==a.NonZeroCount())// a.NonZeroCount()%16==12
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		rowIndices8=B.ColIndices()+rowStart8;
		rowIndices9=B.ColIndices()+rowStart9;
		rowIndices10=B.ColIndices()+rowStart10;
		rowIndices11=B.ColIndices()+rowStart11;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-5==a.NonZeroCount())// a.NonZeroCount()%16==11
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		rowIndices8=B.ColIndices()+rowStart8;
		rowIndices9=B.ColIndices()+rowStart9;
		rowIndices10=B.ColIndices()+rowStart10;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-6==a.NonZeroCount())// a.NonZeroCount()%16==10
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		rowIndices8=B.ColIndices()+rowStart8;
		rowIndices9=B.ColIndices()+rowStart9;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-7==a.NonZeroCount())// a.NonZeroCount()%16==9
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		rowIndices8=B.ColIndices()+rowStart8;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-8==a.NonZeroCount())// a.NonZeroCount()%16==8
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-9==a.NonZeroCount())// a.NonZeroCount()%16==7
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-10==a.NonZeroCount())// a.NonZeroCount()%16==6
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-11==a.NonZeroCount())// a.NonZeroCount()%16==5
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-12==a.NonZeroCount())// a.NonZeroCount()%16==4
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-13==a.NonZeroCount())// a.NonZeroCount()%16==3
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-14==a.NonZeroCount())// a.NonZeroCount()%16==2
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else if(t-15==a.NonZeroCount())// a.NonZeroCount()%16==1
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int rowPos8=0;//Current position into row
	int rowPos9=0;//Current position into row
	int rowPos10=0;//Current position into row
	int rowPos11=0;//Current position into row
	int rowPos12=0;//Current position into row
	int rowPos13=0;//Current position into row
	int rowPos14=0;//Current position into row
	int rowPos15=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	int index8=intMax;
	int index9=intMax;
	int index10=intMax;
	int index11=intMax;
	int index12=intMax;
	int index13=intMax;
	int index14=intMax;
	int index15=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}
	if(rowPos8<rowLength8){
		index8=ldg(rowIndices8+rowPos8);
	}
	if(rowPos9<rowLength9){
		index9=ldg(rowIndices9+rowPos9);
	}
	if(rowPos10<rowLength10){
		index10=ldg(rowIndices10+rowPos10);
	}
	if(rowPos11<rowLength11){
		index11=ldg(rowIndices11+rowPos11);
	}
	if(rowPos12<rowLength12){
		index12=ldg(rowIndices12+rowPos12);
	}
	if(rowPos13<rowLength13){
		index13=ldg(rowIndices13+rowPos13);
	}
	if(rowPos14<rowLength14){
		index14=ldg(rowIndices14+rowPos14);
	}
	if(rowPos15<rowLength15){
		index15=ldg(rowIndices15+rowPos15);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	min_index=index8<min_index?index8:min_index;
	min_index=index9<min_index?index9:min_index;
	min_index=index10<min_index?index10:min_index;
	min_index=index11<min_index?index11:min_index;
	min_index=index12<min_index?index12:min_index;
	min_index=index13<min_index?index13:min_index;
	min_index=index14<min_index?index14:min_index;
	min_index=index15<min_index?index15:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			rowPos1++;
		}
		if(index2==min_index)
		{
			rowPos2++;
		}
		if(index3==min_index)
		{
			rowPos3++;
		}
		if(index4==min_index)
		{
			rowPos4++;
		}
		if(index5==min_index)
		{
			rowPos5++;
		}
		if(index6==min_index)
		{
			rowPos6++;
		}
		if(index7==min_index)
		{
			rowPos7++;
		}
		if(index8==min_index)
		{
			rowPos8++;
		}
		if(index9==min_index)
		{
			rowPos9++;
		}
		if(index10==min_index)
		{
			rowPos10++;
		}
		if(index11==min_index)
		{
			rowPos11++;
		}
		if(index12==min_index)
		{
			rowPos12++;
		}
		if(index13==min_index)
		{
			rowPos13++;
		}
		if(index14==min_index)
		{
			rowPos14++;
		}
		if(index15==min_index)
		{
			rowPos15++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
	//		frontIndex=index0>index1?index1:index0;
	//		frontValue=index0>index1?ldg(rowValues1+rowPos1)*weight1:ldg(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		if(frontIndex==minFront){//put these into tmp and load next elements
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}
			if(rowPos8<rowLength8){
				index8=ldg(rowIndices8+rowPos8);
			}
			else{
				index8=intMax;
			}
			if(rowPos9<rowLength9){
				index9=ldg(rowIndices9+rowPos9);
			}
			else{
				index9=intMax;
			}
			if(rowPos10<rowLength10){
				index10=ldg(rowIndices10+rowPos10);
			}
			else{
				index10=intMax;
			}
			if(rowPos11<rowLength11){
				index11=ldg(rowIndices11+rowPos11);
			}
			else{
				index11=intMax;
			}
			if(rowPos12<rowLength12){
				index12=ldg(rowIndices12+rowPos12);
			}
			else{
				index12=intMax;
			}
			if(rowPos13<rowLength13){
				index13=ldg(rowIndices13+rowPos13);
			}
			else{
				index13=intMax;
			}
			if(rowPos14<rowLength14){
				index14=ldg(rowIndices14+rowPos14);
			}
			else{
				index14=intMax;
			}
			if(rowPos15<rowLength15){
				index15=ldg(rowIndices15+rowPos15);
			}
			else{
				index15=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			min_index=index8<min_index?index8:min_index;
			min_index=index9<min_index?index9:min_index;
			min_index=index10<min_index?index10:min_index;
			min_index=index11<min_index?index11:min_index;
			min_index=index12<min_index?index12:min_index;
			min_index=index13<min_index?index13:min_index;
			min_index=index14<min_index?index14:min_index;
			min_index=index15<min_index?index15:min_index;
			frontIndex=min_index;

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					rowPos1++;
				}
				if(index2==min_index)
				{
					rowPos2++;
				}
				if(index3==min_index)
				{
					rowPos3++;
				}
				if(index4==min_index)
				{
					rowPos4++;
				}
				if(index5==min_index)
				{
					rowPos5++;
				}
				if(index6==min_index)
				{
					rowPos6++;
				}
				if(index7==min_index)
				{
					rowPos7++;
				}
				if(index8==min_index)
				{
					rowPos8++;
				}
				if(index9==min_index)
				{
					rowPos9++;
				}
				if(index10==min_index)
				{
					rowPos10++;
				}
				if(index11==min_index)
				{
					rowPos11++;
				}
				if(index12==min_index)
				{
					rowPos12++;
				}
				if(index13==min_index)
				{
					rowPos13++;
				}
				if(index14==min_index)
				{
					rowPos14++;
				}
				if(index15==min_index)
				{
					rowPos15++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}


		__syncthreads();

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+threadIdx.x]=bufferedIndex;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}
template<int WarpSize, int SegmentSize, typename T>
static __device__ void MulOverWarpValue_16(CSparseVector<T>& dst, CSparseVector<T>& a, CSparseMatrixCSR<T>& B, T *c_val, uint* c_indices){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		CSparseVector<T> b=B.GetRow(a.Index(0));
		for(int i=threadIdx.x;i<dst.NonZeroCount();i+=WarpSize){
			dst.Value(i)=weight*b.Value(i);
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread
	T* rowValues8;uint* rowIndices8;int rowLength8=0;//The row for the thread
	T* rowValues9;uint* rowIndices9;int rowLength9=0;//The row for the thread
	T* rowValues10;uint* rowIndices10;int rowLength10=0;//The row for the thread
	T* rowValues11;uint* rowIndices11;int rowLength11=0;//The row for the thread
	T* rowValues12;uint* rowIndices12;int rowLength12=0;//The row for the thread
	T* rowValues13;uint* rowIndices13;int rowLength13=0;//The row for the thread
	T* rowValues14;uint* rowIndices14;int rowLength14=0;//The row for the thread
	T* rowValues15;uint* rowIndices15;int rowLength15=0;//The row for the thread
	T weight0=0;//The weight for the row
	T weight1=0;//The weight for the row
	T weight2=0;//The weight for the row
	T weight3=0;//The weight for the row
	T weight4=0;//The weight for the row
	T weight5=0;//The weight for the row
	T weight6=0;//The weight for the row
	T weight7=0;//The weight for the row
	T weight8=0;//The weight for the row
	T weight9=0;//The weight for the row
	T weight10=0;//The weight for the row
	T weight11=0;//The weight for the row
	T weight12=0;//The weight for the row
	T weight13=0;//The weight for the row
	T weight14=0;//The weight for the row
	T weight15=0;//The weight for the row
	int t=(threadIdx.x+1)*16;

	if(t<=a.NonZeroCount()){
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint r15=ldg(a.Indices()+d0+15);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		uint rowStart15=ldg(B.RowStarts()+r15);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=ldg(B.RowStarts()+r15+1)-rowStart15;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		rowValues15=B.Values()+rowStart15;
		rowIndices15=B.ColIndices()+rowStart15;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
		weight15=ldg(a.Values()+d0+15);//a.Value(thread);
	}
	else if(t-1==a.NonZeroCount())  //a.NonZeroCount()%16==15
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint r14=ldg(a.Indices()+d0+14);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		uint rowStart14=ldg(B.RowStarts()+r14);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=ldg(B.RowStarts()+r14+1)-rowStart14;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		rowValues14=B.Values()+rowStart14;
		rowIndices14=B.ColIndices()+rowStart14;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
		weight14=ldg(a.Values()+d0+14);//a.Value(thread);
	}
	else if(t-2==a.NonZeroCount()) //a.NonZeroCount()%16==14
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint r13=ldg(a.Indices()+d0+13);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		uint rowStart13=ldg(B.RowStarts()+r13);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=ldg(B.RowStarts()+r13+1)-rowStart13;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		rowValues13=B.Values()+rowStart13;
		rowIndices13=B.ColIndices()+rowStart13;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
		weight13=ldg(a.Values()+d0+13);//a.Value(thread);
	}
	else if(t-3==a.NonZeroCount())// a.NonZeroCount()%16==13
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint r12=ldg(a.Indices()+d0+12);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		uint rowStart12=ldg(B.RowStarts()+r12);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=ldg(B.RowStarts()+r12+1)-rowStart12;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		rowValues12=B.Values()+rowStart12;
		rowIndices12=B.ColIndices()+rowStart12;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
		weight12=ldg(a.Values()+d0+12);//a.Value(thread);
	}
	else if(t-4==a.NonZeroCount())// a.NonZeroCount()%16==12
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint r11=ldg(a.Indices()+d0+11);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		uint rowStart11=ldg(B.RowStarts()+r11);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=ldg(B.RowStarts()+r11+1)-rowStart11;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		rowValues11=B.Values()+rowStart11;
		rowIndices11=B.ColIndices()+rowStart11;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
		weight11=ldg(a.Values()+d0+11);//a.Value(thread);
	}
	else if(t-5==a.NonZeroCount())// a.NonZeroCount()%16==11
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint r10=ldg(a.Indices()+d0+10);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		uint rowStart10=ldg(B.RowStarts()+r10);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=ldg(B.RowStarts()+r10+1)-rowStart10;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		rowValues10=B.Values()+rowStart10;
		rowIndices10=B.ColIndices()+rowStart10;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
		weight10=ldg(a.Values()+d0+10);//a.Value(thread);
	}
	else if(t-6==a.NonZeroCount())// a.NonZeroCount()%16==10
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint r9=ldg(a.Indices()+d0+9);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		uint rowStart9=ldg(B.RowStarts()+r9);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=ldg(B.RowStarts()+r9+1)-rowStart9;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		rowValues9=B.Values()+rowStart9;
		rowIndices9=B.ColIndices()+rowStart9;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
		weight9=ldg(a.Values()+d0+9);//a.Value(thread);
	}
	else if(t-7==a.NonZeroCount())// a.NonZeroCount()%16==9
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint r8=ldg(a.Indices()+d0+8);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		uint rowStart8=ldg(B.RowStarts()+r8);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=ldg(B.RowStarts()+r8+1)-rowStart8;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		rowValues8=B.Values()+rowStart8;
		rowIndices8=B.ColIndices()+rowStart8;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
		weight8=ldg(a.Values()+d0+8);//a.Value(thread);
	}
	else if(t-8==a.NonZeroCount())// a.NonZeroCount()%16==8
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint r7=ldg(a.Indices()+d0+7);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		uint rowStart7=ldg(B.RowStarts()+r7);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=ldg(B.RowStarts()+r7+1)-rowStart7;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		rowValues7=B.Values()+rowStart7;
		rowIndices7=B.ColIndices()+rowStart7;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
		weight7=ldg(a.Values()+d0+7);//a.Value(thread);
	}
	else if(t-9==a.NonZeroCount())// a.NonZeroCount()%16==7
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint r6=ldg(a.Indices()+d0+6);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		uint rowStart6=ldg(B.RowStarts()+r6);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=ldg(B.RowStarts()+r6+1)-rowStart6;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		rowValues6=B.Values()+rowStart6;
		rowIndices6=B.ColIndices()+rowStart6;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
		weight6=ldg(a.Values()+d0+6);//a.Value(thread);
	}
	else if(t-10==a.NonZeroCount())// a.NonZeroCount()%16==6
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint r5=ldg(a.Indices()+d0+5);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		uint rowStart5=ldg(B.RowStarts()+r5);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=ldg(B.RowStarts()+r5+1)-rowStart5;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		rowValues5=B.Values()+rowStart5;
		rowIndices5=B.ColIndices()+rowStart5;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
		weight5=ldg(a.Values()+d0+5);//a.Value(thread);
	}
	else if(t-11==a.NonZeroCount())// a.NonZeroCount()%16==5
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint r4=ldg(a.Indices()+d0+4);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		uint rowStart4=ldg(B.RowStarts()+r4);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=ldg(B.RowStarts()+r4+1)-rowStart4;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		rowValues4=B.Values()+rowStart4;
		rowIndices4=B.ColIndices()+rowStart4;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
		weight4=ldg(a.Values()+d0+4);//a.Value(thread);
	}
	else if(t-12==a.NonZeroCount())// a.NonZeroCount()%16==4
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint r3=ldg(a.Indices()+d0+3);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		uint rowStart3=ldg(B.RowStarts()+r3);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=ldg(B.RowStarts()+r3+1)-rowStart3;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		rowValues3=B.Values()+rowStart3;
		rowIndices3=B.ColIndices()+rowStart3;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
		weight3=ldg(a.Values()+d0+3);//a.Value(thread);
	}
	else if(t-13==a.NonZeroCount())// a.NonZeroCount()%16==3
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint r2=ldg(a.Indices()+d0+2);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		uint rowStart2=ldg(B.RowStarts()+r2);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=ldg(B.RowStarts()+r2+1)-rowStart2;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		rowValues2=B.Values()+rowStart2;
		rowIndices2=B.ColIndices()+rowStart2;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
		weight2=ldg(a.Values()+d0+2);//a.Value(thread);
	}
	else if(t-14==a.NonZeroCount())// a.NonZeroCount()%16==2
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint r1=ldg(a.Indices()+d0+1);
		uint rowStart0=ldg(B.RowStarts()+r0);
		uint rowStart1=ldg(B.RowStarts()+r1);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=ldg(B.RowStarts()+r1+1)-rowStart1;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		rowValues1=B.Values()+rowStart1;
		rowIndices1=B.ColIndices()+rowStart1;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
		weight1=ldg(a.Values()+d0+1);//a.Value(thread);
	}
	else if(t-15==a.NonZeroCount())// a.NonZeroCount()%16==1
	{
		uint d0=threadIdx.x*16;
		uint r0=ldg(a.Indices()+d0);//uint rowIndex=a.Index(thread);		
		uint rowStart0=ldg(B.RowStarts()+r0);
		rowLength0=ldg(B.RowStarts()+r0+1)-rowStart0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
		rowValues0=B.Values()+rowStart0;
		rowIndices0=B.ColIndices()+rowStart0;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight0=ldg(a.Values()+d0);//a.Value(thread);
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	

	int rowPos0=0;//Current position into row
	int rowPos1=0;//Current position into row
	int rowPos2=0;//Current position into row
	int rowPos3=0;//Current position into row
	int rowPos4=0;//Current position into row
	int rowPos5=0;//Current position into row
	int rowPos6=0;//Current position into row
	int rowPos7=0;//Current position into row
	int rowPos8=0;//Current position into row
	int rowPos9=0;//Current position into row
	int rowPos10=0;//Current position into row
	int rowPos11=0;//Current position into row
	int rowPos12=0;//Current position into row
	int rowPos13=0;//Current position into row
	int rowPos14=0;//Current position into row
	int rowPos15=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread

	//in-thread compare
	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;
	int index8=intMax;
	int index9=intMax;
	int index10=intMax;
	int index11=intMax;
	int index12=intMax;
	int index13=intMax;
	int index14=intMax;
	int index15=intMax;
	if(rowPos0<rowLength0){
		index0=ldg(rowIndices0+rowPos0);
	}
	if(rowPos1<rowLength1){
		index1=ldg(rowIndices1+rowPos1);
	}
	if(rowPos2<rowLength2){
		index2=ldg(rowIndices2+rowPos2);
	}
	if(rowPos3<rowLength3){
		index3=ldg(rowIndices3+rowPos3);
	}
	if(rowPos4<rowLength4){
		index4=ldg(rowIndices4+rowPos4);
	}
	if(rowPos5<rowLength5){
		index5=ldg(rowIndices5+rowPos5);
	}
	if(rowPos6<rowLength6){
		index6=ldg(rowIndices6+rowPos6);
	}
	if(rowPos7<rowLength7){
		index7=ldg(rowIndices7+rowPos7);
	}
	if(rowPos8<rowLength8){
		index8=ldg(rowIndices8+rowPos8);
	}
	if(rowPos9<rowLength9){
		index9=ldg(rowIndices9+rowPos9);
	}
	if(rowPos10<rowLength10){
		index10=ldg(rowIndices10+rowPos10);
	}
	if(rowPos11<rowLength11){
		index11=ldg(rowIndices11+rowPos11);
	}
	if(rowPos12<rowLength12){
		index12=ldg(rowIndices12+rowPos12);
	}
	if(rowPos13<rowLength13){
		index13=ldg(rowIndices13+rowPos13);
	}
	if(rowPos14<rowLength14){
		index14=ldg(rowIndices14+rowPos14);
	}
	if(rowPos15<rowLength15){
		index15=ldg(rowIndices15+rowPos15);
	}

	int min_index=index0;

	min_index=index1<min_index?index1:min_index;
	min_index=index2<min_index?index2:min_index;
	min_index=index3<min_index?index3:min_index;
	min_index=index4<min_index?index4:min_index;
	min_index=index5<min_index?index5:min_index;
	min_index=index6<min_index?index6:min_index;
	min_index=index7<min_index?index7:min_index;
	min_index=index8<min_index?index8:min_index;
	min_index=index9<min_index?index9:min_index;
	min_index=index10<min_index?index10:min_index;
	min_index=index11<min_index?index11:min_index;
	min_index=index12<min_index?index12:min_index;
	min_index=index13<min_index?index13:min_index;
	min_index=index14<min_index?index14:min_index;
	min_index=index15<min_index?index15:min_index;
	frontIndex=min_index;

	if(min_index!=intMax)
	{
		if(index0==min_index)
		{
			frontIndex=index0;
			frontValue=ldg(rowValues0+rowPos0)*weight0;
			rowPos0++;
		}
		if(index1==min_index)
		{
			frontValue+=ldg(rowValues1+rowPos1)*weight1;
			rowPos1++;
		}
		if(index2==min_index)
		{
			frontValue+=ldg(rowValues2+rowPos2)*weight2;
			rowPos2++;
		}
		if(index3==min_index)
		{
			frontValue+=ldg(rowValues3+rowPos3)*weight3;
			rowPos3++;
		}
		if(index4==min_index)
		{
			frontValue+=ldg(rowValues4+rowPos4)*weight4;
			rowPos4++;
		}
		if(index5==min_index)
		{
			frontValue+=ldg(rowValues5+rowPos5)*weight5;
			rowPos5++;
		}
		if(index6==min_index)
		{
			frontValue+=ldg(rowValues6+rowPos6)*weight6;
			rowPos6++;
		}
		if(index7==min_index)
		{
			frontValue+=ldg(rowValues7+rowPos7)*weight7;
			rowPos7++;
		}
		if(index8==min_index)
		{
			frontValue+=ldg(rowValues8+rowPos8)*weight8;
			rowPos8++;
		}
		if(index9==min_index)
		{
			frontValue+=ldg(rowValues9+rowPos9)*weight9;
			rowPos9++;
		}
		if(index10==min_index)
		{
			frontValue+=ldg(rowValues10+rowPos10)*weight10;
			rowPos10++;
		}
		if(index11==min_index)
		{
			frontValue+=ldg(rowValues11+rowPos11)*weight11;
			rowPos11++;
		}
		if(index12==min_index)
		{
			frontValue+=ldg(rowValues12+rowPos12)*weight12;
			rowPos12++;
		}
		if(index13==min_index)
		{
			frontValue+=ldg(rowValues13+rowPos13)*weight13;
			rowPos13++;
		}
		if(index14==min_index)
		{
			frontValue+=ldg(rowValues14+rowPos14)*weight14;
			rowPos14++;
		}
		if(index15==min_index)
		{
			frontValue+=ldg(rowValues15+rowPos15)*weight15;
			rowPos15++;
		}
	}
	else
	{
		frontIndex=intMax;
	}
	//		frontIndex=index0>index1?index1:index0;
	//		frontValue=index0>index1?ldg(rowValues1+rowPos1)*weight1:ldg(rowValues0+rowPos0)*weight0;


	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index

	if(laneId==0)
	{
		c_indices[warpId] = minFront;
	}

	__syncthreads();

	minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos0<rowLength0){
				index0=ldg(rowIndices0+rowPos0);
			}
			else{
				index0=intMax;
			}
			if(rowPos1<rowLength1){
				index1=ldg(rowIndices1+rowPos1);
			}
			else{
				index1=intMax;
			}
			if(rowPos2<rowLength2){
				index2=ldg(rowIndices2+rowPos2);
			}
			else{
				index2=intMax;
			}
			if(rowPos3<rowLength3){
				index3=ldg(rowIndices3+rowPos3);
			}
			else{
				index3=intMax;
			}
			if(rowPos4<rowLength4){
				index4=ldg(rowIndices4+rowPos4);
			}
			else{
				index4=intMax;
			}
			if(rowPos5<rowLength5){
				index5=ldg(rowIndices5+rowPos5);
			}
			else{
				index5=intMax;
			}
			if(rowPos6<rowLength6){
				index6=ldg(rowIndices6+rowPos6);
			}
			else{
				index6=intMax;
			}
			if(rowPos7<rowLength7){
				index7=ldg(rowIndices7+rowPos7);
			}
			else{
				index7=intMax;
			}
			if(rowPos8<rowLength8){
				index8=ldg(rowIndices8+rowPos8);
			}
			else{
				index8=intMax;
			}
			if(rowPos9<rowLength9){
				index9=ldg(rowIndices9+rowPos9);
			}
			else{
				index9=intMax;
			}
			if(rowPos10<rowLength10){
				index10=ldg(rowIndices10+rowPos10);
			}
			else{
				index10=intMax;
			}
			if(rowPos11<rowLength11){
				index11=ldg(rowIndices11+rowPos11);
			}
			else{
				index11=intMax;
			}
			if(rowPos12<rowLength12){
				index12=ldg(rowIndices12+rowPos12);
			}
			else{
				index12=intMax;
			}
			if(rowPos13<rowLength13){
				index13=ldg(rowIndices13+rowPos13);
			}
			else{
				index13=intMax;
			}
			if(rowPos14<rowLength14){
				index14=ldg(rowIndices14+rowPos14);
			}
			else{
				index14=intMax;
			}
			if(rowPos15<rowLength15){
				index15=ldg(rowIndices15+rowPos15);
			}
			else{
				index15=intMax;
			}

			min_index=index0;

			min_index=index1<min_index?index1:min_index;
			min_index=index2<min_index?index2:min_index;
			min_index=index3<min_index?index3:min_index;
			min_index=index4<min_index?index4:min_index;
			min_index=index5<min_index?index5:min_index;
			min_index=index6<min_index?index6:min_index;
			min_index=index7<min_index?index7:min_index;
			min_index=index8<min_index?index8:min_index;
			min_index=index9<min_index?index9:min_index;
			min_index=index10<min_index?index10:min_index;
			min_index=index11<min_index?index11:min_index;
			min_index=index12<min_index?index12:min_index;
			min_index=index13<min_index?index13:min_index;
			min_index=index14<min_index?index14:min_index;
			min_index=index15<min_index?index15:min_index;
			frontIndex=min_index;

			frontValue=0;
			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
					frontIndex=index0;
					frontValue=ldg(rowValues0+rowPos0)*weight0;
					rowPos0++;
				}
				if(index1==min_index)
				{
					frontValue+=ldg(rowValues1+rowPos1)*weight1;
					rowPos1++;
				}
				if(index2==min_index)
				{
					frontValue+=ldg(rowValues2+rowPos2)*weight2;
					rowPos2++;
				}
				if(index3==min_index)
				{
					frontValue+=ldg(rowValues3+rowPos3)*weight3;
					rowPos3++;
				}
				if(index4==min_index)
				{
					frontValue+=ldg(rowValues4+rowPos4)*weight4;
					rowPos4++;
				}
				if(index5==min_index)
				{
					frontValue+=ldg(rowValues5+rowPos5)*weight5;
					rowPos5++;
				}
				if(index6==min_index)
				{
					frontValue+=ldg(rowValues6+rowPos6)*weight6;
					rowPos6++;
				}
				if(index7==min_index)
				{
					frontValue+=ldg(rowValues7+rowPos7)*weight7;
					rowPos7++;
				}
				if(index8==min_index)
				{
					frontValue+=ldg(rowValues8+rowPos8)*weight8;
					rowPos8++;
				}
				if(index9==min_index)
				{
					frontValue+=ldg(rowValues9+rowPos9)*weight9;
					rowPos9++;
				}
				if(index10==min_index)
				{
					frontValue+=ldg(rowValues10+rowPos10)*weight10;
					rowPos10++;
				}
				if(index11==min_index)
				{
					frontValue+=ldg(rowValues11+rowPos11)*weight11;
					rowPos11++;
				}
				if(index12==min_index)
				{
					frontValue+=ldg(rowValues12+rowPos12)*weight12;
					rowPos12++;
				}
				if(index13==min_index)
				{
					frontValue+=ldg(rowValues13+rowPos13)*weight13;
					rowPos13++;
				}
				if(index14==min_index)
				{
					frontValue+=ldg(rowValues14+rowPos14)*weight14;
					rowPos14++;
				}
				if(index15==min_index)
				{
					frontValue+=ldg(rowValues15+rowPos15)*weight15;
					rowPos15++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(laneId==0)
		{
			c_val[warpId] = sum;
		}

		__syncthreads();

		sum=(laneId<SegmentSize)?c_val[(warpId/SegmentSize)*SegmentSize+laneId]:0;

		__syncthreads();

		sum=WarpSum<WarpSize>(sum);

		if(threadIdx.x==bufferPos){//Save into buffer
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			c_indices[warpId] = minFront;
		}

		__syncthreads();

		minFront=(laneId<SegmentSize)?c_indices[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		bufferPos++;		
		if(bufferPos==blockDim.x || (minFront==intMax && threadIdx.x<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Values()[dstPos+threadIdx.x]=bufferedValue;
			dstPos+=blockDim.x;
			bufferPos=0;
		}		
	}
}
