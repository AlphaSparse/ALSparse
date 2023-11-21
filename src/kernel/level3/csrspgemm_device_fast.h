#ifndef CSRSPGEMM_DEVICE_FAST
#define CSRSPGEMM_DEVICE_FAST
#include "alphasparse.h"
#include <cub/cub.cuh>  
#include "fast/CSparseMatrixCSR.h"
#include "fast/CSparseVector.h"
#include "fast/CVector.h"
#include "fast/devicehost.h"
#include "fast/DeviceMemBlock.h"
#include "fast/DeviceVector.h"
#include "fast/HostMemBlock.h"
#include "fast/HostVector.h"
#include "fast/Intrinsics.h"
#include "fast/SparseHostVector.h"
#include "fast/SparseHostMatrixCSR.h"
#include "fast/StrideIter.h"
#include "fast/TrackedObject.h"
#include "fast/VectorTypes.h"
#include "fast/Verify.h"
#include "fast/SparseDeviceMatrixCSR.h"
#include "fast/SparseDeviceVector.h"
#include "fast/ldg.h"
#include "fast/WarpReduction.h"
#include "fast/MulWarp.h"

template<typename T>
void ComputeBin(alphasparseSpMatDescr_t A, std::vector<T> &Queue, std::vector<T> &Queue_one){

	T row_num = A->rows;
	std::vector<T> rowLenth(row_num);
	std::vector<T> h_queue(row_num);
	std::vector<T> h_queue_one(14);

    T * h_row_ptr = (T *)malloc(sizeof(T)*(row_num+1));
    cudaMemcpy(h_row_ptr, A->row_data, sizeof(T)*(row_num+1), cudaMemcpyDeviceToHost);

#pragma omp parallel for
    for(int i =0; i < row_num; i++)
        rowLenth[i] = h_row_ptr[i+1] - h_row_ptr[i];
        
	T *queue = h_queue.data();
	T *queue_one = h_queue_one.data();
	
//	uint max_rowlength = 0;

	memset(queue_one, 0, sizeof(T)*14);

	for(T i=0;i<rowLenth.size();i++){
//		if(rowLenth[i]>max_rowlength)
//			max_rowlength = rowLenth[i];
		if(rowLenth[i]<=2) queue_one[0]++;
		else if(rowLenth[i]<=4) queue_one[1]++;
		else if(rowLenth[i]<=8) queue_one[2]++;
		else if(rowLenth[i]<=16) queue_one[3]++;
		else if(rowLenth[i]<=32) queue_one[4]++;
		else if(rowLenth[i]<=64) queue_one[5]++;
		else if(rowLenth[i]<=128) queue_one[6]++;
		else if(rowLenth[i]<=256) queue_one[7]++;
		else if(rowLenth[i]<=512) queue_one[8]++;
		else if(rowLenth[i]<=1024) queue_one[9]++;
		else if(rowLenth[i]<=2048) queue_one[10]++;
		else if(rowLenth[i]<=4096) queue_one[11]++;
		else queue_one[12]++;
	}

//	std::cout << "Max RowLength" << max_rowlength << std::endl;

	int old_val, new_val;
	old_val = queue_one[0];
	queue_one[0] = 0;

	for(T i=1; i<14; i++)
	{
		new_val = queue_one[i];
		queue_one[i] = old_val + queue_one[i-1];
		old_val = new_val;
	}


	int count2_temp=0,count4_temp=queue_one[1],count8_temp=queue_one[2],count16_temp=queue_one[3],count32_temp=queue_one[4],count64_temp=queue_one[5],count128_temp=queue_one[6],count256_temp=queue_one[7],count512_temp=queue_one[8],count1024_temp=queue_one[9],count2048_temp=queue_one[10],count4096_temp=queue_one[11],countlast_temp=queue_one[12];
	for(T i=0;i<row_num;i++){
		if(rowLenth[i]<=2)
		{
			queue[count2_temp++]=i;
		}
		else if(rowLenth[i]<=4)
		{
			queue[count4_temp++]=i;
		}
		else if(rowLenth[i]<=8)
		{
			queue[count8_temp++]=i;
		}
		else if(rowLenth[i]<=16)
		{
			queue[count16_temp++]=i;
		}
		else if(rowLenth[i]<=32)
		{
			queue[count32_temp++]=i;
		}
		else if(rowLenth[i]<=64)
		{
			queue[count64_temp++]=i;
		}
		else if(rowLenth[i]<=128)
		{
			queue[count128_temp++]=i;
		}
		else if(rowLenth[i]<=256)
		{
			queue[count256_temp++]=i;
		}
		else if(rowLenth[i]<=512)
		{
			queue[count512_temp++]=i;
		}
		else if(rowLenth[i]<=1024)
		{
			queue[count1024_temp++]=i;
		}
		else if(rowLenth[i]<=2048)
		{
			queue[count2048_temp++]=i;
		}
		else if(rowLenth[i]<=4096)
		{
			queue[count4096_temp++]=i;
		}
		else if(rowLenth[i]>4096)
		{
			queue[countlast_temp++]=i;
		}
	}
//	std::cout << "queue[" <<count256 << "] = " << queue[count256] << std::endl;
//	std::cout << "rowLenth[" << queue[count256] << "]" << rowLenth[queue[count256]] << std::endl;
	Queue = h_queue;
	Queue_one = h_queue_one;
}

template<int WarpSize, typename T>
static __device__ uint MulWarpPredictSize(CSparseVector<T> rowA, CSparseMatrixCSR<T> B){
	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));
	const int intMax=B.Width();	
	T* rowValues;uint* rowIndices;int rowLength=0;//The row for the thread	
	if(threadIdx.x<rowA.NonZeroCount())
		B.GetRow(rowA.Index(threadIdx.x),rowValues,rowIndices,rowLength);

	int rowPos=0;//position into row
	int frontIndex=intMax;//Means that the row ended
	if(rowPos<rowLength){
		frontIndex=ldg(rowIndices+rowPos);		
		rowPos++;
	}
	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
			//load next
			if(rowPos<rowLength){				
				frontIndex=(int)ldg(rowIndices+rowPos);
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		dstPos++;
	}
	return dstPos;
}

//Similar to MulWarp but only computes the size.
template<int WarpSize, typename T>
static __device__ uint MulWarpPredictSize_2(CSparseVector<T> rowA, CSparseMatrixCSR<T> B){
	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread	

	int t=(threadIdx.x+1)*2;
	
	if(t<=rowA.NonZeroCount())
	{
		B.GetRow(rowA.Index(threadIdx.x*2),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*2+1),rowValues1,rowIndices1,rowLength1);
	}
	else if(t-1==rowA.NonZeroCount())
	{
		B.GetRow(rowA.Index(threadIdx.x*2),rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
	}


	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int frontIndex=intMax;//Means that the row ended

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
		rowPos0++;
	}
	else if(index0>index1)
	{
		frontIndex=index1;
		rowPos1++;
	}
	else
	{
		if(index0!=intMax)
		{
			frontIndex=index0;
			rowPos0++;
			rowPos1++;
		}
		else
		{
		}
	}

	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
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
				rowPos0++;
			}
			else if(index0>index1)
			{
				frontIndex=index1;
				rowPos1++;
			}
			else
			{
				if(index0!=intMax)
				{
					frontIndex=index0;
					rowPos0++;
					rowPos1++;
				}
				else
				{
					frontIndex=intMax;
				}
			}

		}
		minFront=WarpMin<WarpSize>(frontIndex);
		dstPos++;
	}
	return dstPos;
}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, typename T>
static __device__ uint MulWarpPredictSize_4(CSparseVector<T> rowA, CSparseMatrixCSR<T> B){
	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread	
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread	
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread	

	int t=(threadIdx.x+1)*4;
	if(t<=rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*4),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*4+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*4+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*4+3),rowValues3,rowIndices3,rowLength3);
	}
	else if(t-1==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*4),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*4+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*4+2),rowValues2,rowIndices2,rowLength2);
		rowLength3=0;
	}
	else if(t-2==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*4),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*4+1),rowValues1,rowIndices1,rowLength1);
		rowLength2=0;
		rowLength3=0;
	}
	else if(t-3==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*4),rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}
	else{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row


	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;

	int frontIndex=intMax;//Means that the row ended

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
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;
//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0&&threadIdx.y==0)
//		{
//			printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//			printf("index0=%d,index1=%d,index2=%d,index3=%d\n",index0,index1,index2,index3);
//			printf("frontIndex=%d\n",frontIndex);
//			printf("minFront=%d\n",minFront);
//			printf("------------------------------------\n");
//		}
//	}
	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
			}
			else
			{
				frontIndex=intMax;
			}
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		dstPos++;

	}

	return dstPos;
}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, typename T>
static __device__ uint MulWarpPredictSize_8(CSparseVector<T> rowA, CSparseMatrixCSR<T> B){
	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread	
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread	
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread	
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread	
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread	
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread	
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread	

	int t=(threadIdx.x+1)*8;
	if(t<=rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*8+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*8+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*8+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*8+7),rowValues7,rowIndices7,rowLength7);
	}
	else if(t-1==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*8+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*8+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*8+6),rowValues6,rowIndices6,rowLength6);
		rowLength7=0;
	}
	else if(t-2==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*8+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*8+5),rowValues5,rowIndices5,rowLength5);
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-3==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*8+4),rowValues4,rowIndices4,rowLength4);
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-4==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-5==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-6==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-7==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row


	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;

	int frontIndex=intMax;//Means that the row ended

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
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;
//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0&&threadIdx.y==0)
//		{
//			printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//			printf("index0=%d,index1=%d,index2=%d,index3=%d\n",index0,index1,index2,index3);
//			printf("frontIndex=%d\n",frontIndex);
//			printf("minFront=%d\n",minFront);
//			printf("------------------------------------\n");
//		}
//	}
	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
			}
			else
			{
				frontIndex=intMax;
			}
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		dstPos++;

	}

	return dstPos;
}
//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, typename T>
static __device__ uint MulWarpPredictSize_16(CSparseVector<T> rowA, CSparseMatrixCSR<T> B){
	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

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

	int t=(threadIdx.x+1)*16;
	if(t<=rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*16+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*16+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*16+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*16+15),rowValues15,rowIndices15,rowLength15);
	}
	else if(t-1==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*16+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*16+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*16+14),rowValues14,rowIndices14,rowLength14);
		rowLength15=0;
	}
	else if(t-2==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*16+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*16+13),rowValues13,rowIndices13,rowLength13);
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-3==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*16+12),rowValues12,rowIndices12,rowLength12);
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-4==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-5==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-6==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-7==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-8==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-9==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
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
	else if(t-10==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
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
	else if(t-11==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
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
	else if(t-12==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
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
	else if(t-13==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
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
	else if(t-14==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
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
	else if(t-15==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
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
	else{
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
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row
	int rowPos8=0;//position into row
	int rowPos9=0;//position into row
	int rowPos10=0;//position into row
	int rowPos11=0;//position into row
	int rowPos12=0;//position into row
	int rowPos13=0;//position into row
	int rowPos14=0;//position into row
	int rowPos15=0;//position into row


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

	int frontIndex=intMax;//Means that the row ended

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



	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;
//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0&&threadIdx.y==0)
//		{
//			printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//			printf("index0=%d,index1=%d,index2=%d,index3=%d\n",index0,index1,index2,index3);
//			printf("frontIndex=%d\n",frontIndex);
//			printf("minFront=%d\n",minFront);
//			printf("------------------------------------\n");
//		}
//	}
	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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
				//	frontIndex=index0;
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
		minFront=WarpMin<WarpSize>(frontIndex);

		dstPos++;

	}

	return dstPos;
}


//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, typename T>
static __device__ uint MulWarpPredictSize_32(CSparseVector<T> rowA, CSparseMatrixCSR<T> B){
	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

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

	int t=(threadIdx.x+1)*32;
	if(t<=rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		B.GetRow(rowA.Index(threadIdx.x*32+24),rowValues24,rowIndices24,rowLength24);
		B.GetRow(rowA.Index(threadIdx.x*32+25),rowValues25,rowIndices25,rowLength25);
		B.GetRow(rowA.Index(threadIdx.x*32+26),rowValues26,rowIndices26,rowLength26);
		B.GetRow(rowA.Index(threadIdx.x*32+27),rowValues27,rowIndices27,rowLength27);
		B.GetRow(rowA.Index(threadIdx.x*32+28),rowValues28,rowIndices28,rowLength28);
		B.GetRow(rowA.Index(threadIdx.x*32+29),rowValues29,rowIndices29,rowLength29);
		B.GetRow(rowA.Index(threadIdx.x*32+30),rowValues30,rowIndices30,rowLength30);
		B.GetRow(rowA.Index(threadIdx.x*32+31),rowValues31,rowIndices31,rowLength31);
	}
	else if(t-1==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		B.GetRow(rowA.Index(threadIdx.x*32+24),rowValues24,rowIndices24,rowLength24);
		B.GetRow(rowA.Index(threadIdx.x*32+25),rowValues25,rowIndices25,rowLength25);
		B.GetRow(rowA.Index(threadIdx.x*32+26),rowValues26,rowIndices26,rowLength26);
		B.GetRow(rowA.Index(threadIdx.x*32+27),rowValues27,rowIndices27,rowLength27);
		B.GetRow(rowA.Index(threadIdx.x*32+28),rowValues28,rowIndices28,rowLength28);
		B.GetRow(rowA.Index(threadIdx.x*32+29),rowValues29,rowIndices29,rowLength29);
		B.GetRow(rowA.Index(threadIdx.x*32+30),rowValues30,rowIndices30,rowLength30);
		rowLength31=0;
	}
	else if(t-2==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		B.GetRow(rowA.Index(threadIdx.x*32+24),rowValues24,rowIndices24,rowLength24);
		B.GetRow(rowA.Index(threadIdx.x*32+25),rowValues25,rowIndices25,rowLength25);
		B.GetRow(rowA.Index(threadIdx.x*32+26),rowValues26,rowIndices26,rowLength26);
		B.GetRow(rowA.Index(threadIdx.x*32+27),rowValues27,rowIndices27,rowLength27);
		B.GetRow(rowA.Index(threadIdx.x*32+28),rowValues28,rowIndices28,rowLength28);
		B.GetRow(rowA.Index(threadIdx.x*32+29),rowValues29,rowIndices29,rowLength29);
		rowLength30=0;
		rowLength31=0;
	}
	else if(t-3==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		B.GetRow(rowA.Index(threadIdx.x*32+24),rowValues24,rowIndices24,rowLength24);
		B.GetRow(rowA.Index(threadIdx.x*32+25),rowValues25,rowIndices25,rowLength25);
		B.GetRow(rowA.Index(threadIdx.x*32+26),rowValues26,rowIndices26,rowLength26);
		B.GetRow(rowA.Index(threadIdx.x*32+27),rowValues27,rowIndices27,rowLength27);
		B.GetRow(rowA.Index(threadIdx.x*32+28),rowValues28,rowIndices28,rowLength28);
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
	}
	else if(t-4==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		B.GetRow(rowA.Index(threadIdx.x*32+24),rowValues24,rowIndices24,rowLength24);
		B.GetRow(rowA.Index(threadIdx.x*32+25),rowValues25,rowIndices25,rowLength25);
		B.GetRow(rowA.Index(threadIdx.x*32+26),rowValues26,rowIndices26,rowLength26);
		B.GetRow(rowA.Index(threadIdx.x*32+27),rowValues27,rowIndices27,rowLength27);
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
	}
	else if(t-5==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		B.GetRow(rowA.Index(threadIdx.x*32+24),rowValues24,rowIndices24,rowLength24);
		B.GetRow(rowA.Index(threadIdx.x*32+25),rowValues25,rowIndices25,rowLength25);
		B.GetRow(rowA.Index(threadIdx.x*32+26),rowValues26,rowIndices26,rowLength26);
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
	}
	else if(t-6==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		B.GetRow(rowA.Index(threadIdx.x*32+24),rowValues24,rowIndices24,rowLength24);
		B.GetRow(rowA.Index(threadIdx.x*32+25),rowValues25,rowIndices25,rowLength25);
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
	}
	else if(t-7==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		B.GetRow(rowA.Index(threadIdx.x*32+24),rowValues24,rowIndices24,rowLength24);
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
	}
	else if(t-8==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
		B.GetRow(rowA.Index(threadIdx.x*32+23),rowValues23,rowIndices23,rowLength23);
		rowLength24=0;
		rowLength25=0;
		rowLength26=0;
		rowLength27=0;
		rowLength28=0;
		rowLength29=0;
		rowLength30=0;
		rowLength31=0;
	}
	else if(t-9==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
		B.GetRow(rowA.Index(threadIdx.x*32+22),rowValues22,rowIndices22,rowLength22);
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
	else if(t-10==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
		B.GetRow(rowA.Index(threadIdx.x*32+21),rowValues21,rowIndices21,rowLength21);
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
	else if(t-11==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
		B.GetRow(rowA.Index(threadIdx.x*32+20),rowValues20,rowIndices20,rowLength20);
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
	else if(t-12==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
		B.GetRow(rowA.Index(threadIdx.x*32+19),rowValues19,rowIndices19,rowLength19);
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
	else if(t-13==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
		B.GetRow(rowA.Index(threadIdx.x*32+18),rowValues18,rowIndices18,rowLength18);
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
	else if(t-14==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
		B.GetRow(rowA.Index(threadIdx.x*32+17),rowValues17,rowIndices17,rowLength17);
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
	else if(t-15==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
		B.GetRow(rowA.Index(threadIdx.x*32+16),rowValues16,rowIndices16,rowLength16);
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
	else if(t-16==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*32+15),rowValues15,rowIndices15,rowLength15);
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
	else if(t-17==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*32+14),rowValues14,rowIndices14,rowLength14);
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
	else if(t-18==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*32+13),rowValues13,rowIndices13,rowLength13);
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
	else if(t-19==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*32+12),rowValues12,rowIndices12,rowLength12);
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
	else if(t-20==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*32+11),rowValues11,rowIndices11,rowLength11);
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
	else if(t-21==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*32+10),rowValues10,rowIndices10,rowLength10);
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
	else if(t-22==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*32+9),rowValues9,rowIndices9,rowLength9);
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
	else if(t-23==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*32+8),rowValues8,rowIndices8,rowLength8);
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
	else if(t-24==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*32+7),rowValues7,rowIndices7,rowLength7);
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
	else if(t-25==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*32+6),rowValues6,rowIndices6,rowLength6);
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
	else if(t-26==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*32+5),rowValues5,rowIndices5,rowLength5);
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
	else if(t-27==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*32+4),rowValues4,rowIndices4,rowLength4);
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
	else if(t-28==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*32+3),rowValues3,rowIndices3,rowLength3);
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
	else if(t-29==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*32+2),rowValues2,rowIndices2,rowLength2);
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
	else if(t-30==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*32+1),rowValues1,rowIndices1,rowLength1);
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
	else if(t-31==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*32),rowValues0,rowIndices0,rowLength0);
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
	else{
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
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row
	int rowPos8=0;//position into row
	int rowPos9=0;//position into row
	int rowPos10=0;//position into row
	int rowPos11=0;//position into row
	int rowPos12=0;//position into row
	int rowPos13=0;//position into row
	int rowPos14=0;//position into row
	int rowPos15=0;//position into row
	int rowPos16=0;//position into row
	int rowPos17=0;//position into row
	int rowPos18=0;//position into row
	int rowPos19=0;//position into row
	int rowPos20=0;//position into row
	int rowPos21=0;//position into row
	int rowPos22=0;//position into row
	int rowPos23=0;//position into row
	int rowPos24=0;//position into row
	int rowPos25=0;//position into row
	int rowPos26=0;//position into row
	int rowPos27=0;//position into row
	int rowPos28=0;//position into row
	int rowPos29=0;//position into row
	int rowPos30=0;//position into row
	int rowPos31=0;//position into row


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

	int frontIndex=intMax;//Means that the row ended

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
		if(index16==min_index)
		{
			rowPos16++;
		}
		if(index17==min_index)
		{
			rowPos17++;
		}
		if(index18==min_index)
		{
			rowPos18++;
		}
		if(index19==min_index)
		{
			rowPos19++;
		}
		if(index20==min_index)
		{
			rowPos20++;
		}
		if(index21==min_index)
		{
			rowPos21++;
		}
		if(index22==min_index)
		{
			rowPos22++;
		}
		if(index23==min_index)
		{
			rowPos23++;
		}
		if(index24==min_index)
		{
			rowPos24++;
		}
		if(index25==min_index)
		{
			rowPos25++;
		}
		if(index26==min_index)
		{
			rowPos26++;
		}
		if(index27==min_index)
		{
			rowPos27++;
		}
		if(index28==min_index)
		{
			rowPos28++;
		}
		if(index29==min_index)
		{
			rowPos29++;
		}
		if(index30==min_index)
		{
			rowPos30++;
		}
		if(index31==min_index)
		{
			rowPos31++;
		}
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;
//	if(blockIdx.x==0)
//	{
//		if(threadIdx.x==0&&threadIdx.y==0)
//		{
//			printf("threadIdx.x=%d,threadIdx.y=%d\n",threadIdx.x,threadIdx.y);
//			printf("index0=%d,index1=%d,index2=%d,index3=%d\n",index0,index1,index2,index3);
//			printf("frontIndex=%d\n",frontIndex);
//			printf("minFront=%d\n",minFront);
//			printf("------------------------------------\n");
//		}
//	}
	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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
			if(rowPos27<rowLength15){
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
				if(index16==min_index)
				{
					rowPos16++;
				}
				if(index17==min_index)
				{
					rowPos17++;
				}
				if(index18==min_index)
				{
					rowPos18++;
				}
				if(index19==min_index)
				{
					rowPos19++;
				}
				if(index20==min_index)
				{
					rowPos20++;
				}
				if(index21==min_index)
				{
					rowPos21++;
				}
				if(index22==min_index)
				{
					rowPos22++;
				}
				if(index23==min_index)
				{
					rowPos23++;
				}
				if(index24==min_index)
				{
					rowPos24++;
				}
				if(index25==min_index)
				{
					rowPos25++;
				}
				if(index26==min_index)
				{
					rowPos26++;
				}
				if(index27==min_index)
				{
					rowPos27++;
				}
				if(index28==min_index)
				{
					rowPos28++;
				}
				if(index29==min_index)
				{
					rowPos29++;
				}
				if(index30==min_index)
				{
					rowPos30++;
				}
				if(index31==min_index)
				{
					rowPos31++;
				}
			}
			else
			{
				frontIndex=intMax;
			}
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		dstPos++;

	}

	return dstPos;
}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ uint MulOverWarpPredictSize_1(CSparseVector<T> rowA, CSparseMatrixCSR<T> B,uint *temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues;uint* rowIndices;int rowLength=0;//The row for the thread	
	if(threadIdx.x<rowA.NonZeroCount())
		B.GetRow(rowA.Index(threadIdx.x),rowValues,rowIndices,rowLength);

	int rowPos=0;//position into row
	int frontIndex=intMax;//Means that the row ended
	if(rowPos<rowLength){
		frontIndex=ldg(rowIndices+rowPos);		
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);	

	if(laneId==0)
	{
		temp[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
			//load next
			if(rowPos<rowLength){				
				frontIndex=(int)ldg(rowIndices+rowPos);
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;
	}
	return dstPos;

}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ uint MulOverWarpPredictSize_2(CSparseVector<T> rowA, CSparseMatrixCSR<T> B,uint *temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread	

	int t=(threadIdx.x+1)*2;
	
	if(t<=rowA.NonZeroCount())
	{
		B.GetRow(rowA.Index(threadIdx.x*2),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*2+1),rowValues1,rowIndices1,rowLength1);
	}
	else if(t-1==rowA.NonZeroCount())
	{
		B.GetRow(rowA.Index(threadIdx.x*2),rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
	}
	else
	{
		rowLength0=0;
		rowLength1=0;
	}


	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int frontIndex=intMax;//Means that the row ended

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
		rowPos0++;
	}
	else if(index0>index1)
	{
		frontIndex=index1;
		rowPos1++;
	}
	else
	{
		if(index0!=intMax)
		{
			frontIndex=index0;
			rowPos0++;
			rowPos1++;
		}
		else
		{
		}
	}

	int minFront=WarpMin<WarpSize>(frontIndex);	

	if(laneId==0)
	{
		temp[warpId] = minFront;
	}
	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;
	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
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
				rowPos0++;
			}
			else if(index0>index1)
			{
				frontIndex=index1;
				rowPos1++;
			}
			else
			{
				if(index0!=intMax)
				{
					frontIndex=index0;
					rowPos0++;
					rowPos1++;
				}
				else
				{
					frontIndex=intMax;
				}
			}

		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;
	}
	return dstPos;
}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ uint MulOverWarpPredictSize_4(CSparseVector<T> rowA, CSparseMatrixCSR<T> B,uint *temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread	
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread	
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread	

	int t=(threadIdx.x+1)*4;
	if(t<=rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*4),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*4+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*4+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*4+3),rowValues3,rowIndices3,rowLength3);
	}
	else if(t-1==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*4),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*4+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*4+2),rowValues2,rowIndices2,rowLength2);
		rowLength3=0;
	}
	else if(t-2==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*4),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*4+1),rowValues1,rowIndices1,rowLength1);
		rowLength2=0;
		rowLength3=0;
	}
	else if(t-3==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*4),rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}
	else{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
	}
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row


	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;

	int frontIndex=intMax;//Means that the row ended

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
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	

	if(laneId==0)
	{
		temp[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;
	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
			}
			else
			{
				frontIndex=intMax;
			}
		}
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;
		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;

	}

	return dstPos;

}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ uint MulOverWarpPredictSize_8(CSparseVector<T> rowA, CSparseMatrixCSR<T> B,uint *temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

	const int intMax=2147483647;//used to signal that a row is finished
	T* rowValues0;uint* rowIndices0;int rowLength0=0;//The row for the thread	
	T* rowValues1;uint* rowIndices1;int rowLength1=0;//The row for the thread	
	T* rowValues2;uint* rowIndices2;int rowLength2=0;//The row for the thread	
	T* rowValues3;uint* rowIndices3;int rowLength3=0;//The row for the thread	
	T* rowValues4;uint* rowIndices4;int rowLength4=0;//The row for the thread	
	T* rowValues5;uint* rowIndices5;int rowLength5=0;//The row for the thread	
	T* rowValues6;uint* rowIndices6;int rowLength6=0;//The row for the thread	
	T* rowValues7;uint* rowIndices7;int rowLength7=0;//The row for the thread	

	int t=(threadIdx.x+1)*8;
	if(t<=rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*8+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*8+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*8+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*8+7),rowValues7,rowIndices7,rowLength7);
	}
	else if(t-1==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*8+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*8+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*8+6),rowValues6,rowIndices6,rowLength6);
		rowLength7=0;
	}
	else if(t-2==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*8+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*8+5),rowValues5,rowIndices5,rowLength5);
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-3==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*8+4),rowValues4,rowIndices4,rowLength4);
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-4==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*8+3),rowValues3,rowIndices3,rowLength3);
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-5==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*8+2),rowValues2,rowIndices2,rowLength2);
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-6==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*8+1),rowValues1,rowIndices1,rowLength1);
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else if(t-7==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*8),rowValues0,rowIndices0,rowLength0);
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	else{
		rowLength0=0;
		rowLength1=0;
		rowLength2=0;
		rowLength3=0;
		rowLength4=0;
		rowLength5=0;
		rowLength6=0;
		rowLength7=0;
	}
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row


	int index0=intMax;
	int index1=intMax;
	int index2=intMax;
	int index3=intMax;
	int index4=intMax;
	int index5=intMax;
	int index6=intMax;
	int index7=intMax;

	int frontIndex=intMax;//Means that the row ended

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
	}
	else
	{
		frontIndex=intMax;
	}



	int minFront=WarpMin<WarpSize>(frontIndex);	
	if(laneId==0)
	{
		temp[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);
	int dstPos=0;

	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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

			if(min_index!=intMax)
			{
				if(index0==min_index)
				{
				//	frontIndex=index0;
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
			}
			else
			{
				frontIndex=intMax;
			}
		}

		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;

	}

	return dstPos;

}

//***************************************************************************************
//Similar to MulWarp but only computes the size.
template<int WarpSize, int SegmentSize, typename T>
static __device__ uint MulOverWarpPredictSize_16(CSparseVector<T> rowA, CSparseMatrixCSR<T> B,uint *temp){

	int laneId = threadIdx.x & 0x1f;
	int warpId = (threadIdx.x+threadIdx.y*blockDim.x)/32;

	if(rowA.NonZeroCount()==0)
		return 0;
	if(rowA.NonZeroCount()==1)
		return B.RowLength(rowA.Index(0));

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

	int t=(threadIdx.x+1)*16;
	if(t<=rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*16+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*16+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*16+14),rowValues14,rowIndices14,rowLength14);
		B.GetRow(rowA.Index(threadIdx.x*16+15),rowValues15,rowIndices15,rowLength15);
	}
	else if(t-1==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*16+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*16+13),rowValues13,rowIndices13,rowLength13);
		B.GetRow(rowA.Index(threadIdx.x*16+14),rowValues14,rowIndices14,rowLength14);
		rowLength15=0;
	}
	else if(t-2==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*16+12),rowValues12,rowIndices12,rowLength12);
		B.GetRow(rowA.Index(threadIdx.x*16+13),rowValues13,rowIndices13,rowLength13);
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-3==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		B.GetRow(rowA.Index(threadIdx.x*16+12),rowValues12,rowIndices12,rowLength12);
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-4==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		B.GetRow(rowA.Index(threadIdx.x*16+11),rowValues11,rowIndices11,rowLength11);
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-5==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		B.GetRow(rowA.Index(threadIdx.x*16+10),rowValues10,rowIndices10,rowLength10);
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-6==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		B.GetRow(rowA.Index(threadIdx.x*16+9),rowValues9,rowIndices9,rowLength9);
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-7==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		B.GetRow(rowA.Index(threadIdx.x*16+8),rowValues8,rowIndices8,rowLength8);
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-8==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
		B.GetRow(rowA.Index(threadIdx.x*16+7),rowValues7,rowIndices7,rowLength7);
		rowLength8=0;
		rowLength9=0;
		rowLength10=0;
		rowLength11=0;
		rowLength12=0;
		rowLength13=0;
		rowLength14=0;
		rowLength15=0;
	}
	else if(t-9==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
		B.GetRow(rowA.Index(threadIdx.x*16+6),rowValues6,rowIndices6,rowLength6);
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
	else if(t-10==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
		B.GetRow(rowA.Index(threadIdx.x*16+5),rowValues5,rowIndices5,rowLength5);
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
	else if(t-11==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
		B.GetRow(rowA.Index(threadIdx.x*16+4),rowValues4,rowIndices4,rowLength4);
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
	else if(t-12==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
		B.GetRow(rowA.Index(threadIdx.x*16+3),rowValues3,rowIndices3,rowLength3);
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
	else if(t-13==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
		B.GetRow(rowA.Index(threadIdx.x*16+2),rowValues2,rowIndices2,rowLength2);
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
	else if(t-14==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
		B.GetRow(rowA.Index(threadIdx.x*16+1),rowValues1,rowIndices1,rowLength1);
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
	else if(t-15==rowA.NonZeroCount()){
		B.GetRow(rowA.Index(threadIdx.x*16),rowValues0,rowIndices0,rowLength0);
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
	else{
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
	int rowPos0=0;//position into row
	int rowPos1=0;//position into row
	int rowPos2=0;//position into row
	int rowPos3=0;//position into row
	int rowPos4=0;//position into row
	int rowPos5=0;//position into row
	int rowPos6=0;//position into row
	int rowPos7=0;//position into row
	int rowPos8=0;//position into row
	int rowPos9=0;//position into row
	int rowPos10=0;//position into row
	int rowPos11=0;//position into row
	int rowPos12=0;//position into row
	int rowPos13=0;//position into row
	int rowPos14=0;//position into row
	int rowPos15=0;//position into row


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

	int frontIndex=intMax;//Means that the row ended

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



	int minFront=WarpMin<WarpSize>(frontIndex);	

	if(laneId==0)
	{
		temp[warpId] = minFront;
	}

	__syncthreads();

	minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

	__syncthreads();

	minFront=WarpMin<WarpSize>(minFront);

	int dstPos=0;

	while(minFront!=intMax)
	{		
		if(frontIndex==minFront){			
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
				//	frontIndex=index0;
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
		minFront=WarpMin<WarpSize>(frontIndex);

		if(laneId==0)
		{
			temp[warpId] = minFront;
		}
		__syncthreads();

		minFront = (laneId < SegmentSize)? temp[(warpId/SegmentSize)*SegmentSize+laneId]:intMax;

		__syncthreads();

		minFront=WarpMin<WarpSize>(minFront);

		dstPos++;

	}

	return dstPos;
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_1(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	uint dstLength=MulWarpPredictSize<WarpSize>(A.GetRow(r), B);
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_2(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	uint dstLength=MulWarpPredictSize_2<WarpSize>(A.GetRow(r), B); uint *data = Crowstarts.Data(); if(threadIdx.x==0) { data[r] = dstLength;
	}
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_4(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	uint dstLength=MulWarpPredictSize_4<WarpSize>(A.GetRow(r), B);
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_8(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	uint dstLength=MulWarpPredictSize_8<WarpSize>(A.GetRow(r), B);
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeWarpKernel_16(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	uint dstLength=MulWarpPredictSize_16<WarpSize>(A.GetRow(r), B);
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_1(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ uint temp[]; 


	uint dstLength=MulOverWarpPredictSize_1<WarpSize,SegmentSize>(A.GetRow(r), B, temp);

	
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_2(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ uint temp[]; 


	uint dstLength=MulOverWarpPredictSize_2<WarpSize,SegmentSize>(A.GetRow(r), B, temp);

	
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_4(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ uint temp[]; 


	uint dstLength=MulOverWarpPredictSize_4<WarpSize,SegmentSize>(A.GetRow(r), B, temp);

	
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_8(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{
	
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ uint temp[]; 


	uint dstLength=MulOverWarpPredictSize_8<WarpSize,SegmentSize>(A.GetRow(r), B, temp);

	
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}


template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmPredictSizeOverWarpKernel_16(CVector<uint> Crowstarts, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CVector<uint> Queue, CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	extern __shared__ uint temp[]; 


	uint dstLength=MulOverWarpPredictSize_16<WarpSize,SegmentSize>(A.GetRow(r), B, temp);

	
	uint *data = Crowstarts.Data();
	if(threadIdx.x==0)
	{
		data[r] = dstLength;
	}
}

template<typename T, typename U>
void __cdecl PredictCSize(DeviceVector<uint> CRowStarts, SparseDeviceMatrixCSR<U> A, SparseDeviceMatrixCSR<U> B, DeviceVector<uint> Queue, DeviceVector<uint> Queue_one, T* h_queue_one)
{
	int threadnum = 128;

	cudaStream_t stream[13];
	for(int i=0; i<13; i++)
	{
		cudaStreamCreate(&stream[i]);
	}

	uint count;

	for(int i=0; i<13; i++)
	{
		count = h_queue_one[i+1] - h_queue_one[i];
		if(count==0)
			continue;

		if(i==0) //rowLength<=2
		{
			dim3 blockDim(2,threadnum/2,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeWarpKernel_1<2> <<< gridDim, blockDim, 0, stream[0]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==1) //2<rowLength<=4
		{
			dim3 blockDim(4,threadnum/4,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeWarpKernel_1<4> <<< gridDim, blockDim, 0, stream[1]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==2) //5<rowLength<=8
		{
			dim3 blockDim(8,threadnum/8,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeWarpKernel_1<8> <<< gridDim, blockDim, 0, stream[2]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==3) //9<rowLength<=16
		{
			dim3 blockDim(1,threadnum/1,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeWarpKernel_16<1> <<< gridDim, blockDim, 0, stream[3]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==4) //16<rowLength<=32
		{
			dim3 blockDim(32,threadnum/32,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeWarpKernel_1<32> <<< gridDim, blockDim, 0, stream[4]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==5) //32<rowLeng<=64
		{
			dim3 blockDim(32,threadnum/32,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeWarpKernel_2<32> <<< gridDim, blockDim, 0, stream[5]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
//			DifSpmmPredictSizeOverWarpKernel_1<32,2> <<<gridDim, blockDim, threadnum/32, stream[9]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==6) //64<rowLength<=128
		{
			dim3 blockDim(32,8,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeWarpKernel_4<32> <<< gridDim, blockDim, 0, stream[6]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
//			DifSpmmPredictSizeOverWarpKernel_2<32,2> <<<gridDim, blockDim, threadnum/32, stream[9]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==7) //128<rowLength<=256
		{
			dim3 blockDim(32,4,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
//			DifSpmmPredictSizeOverWarpKernel_4<32,2> <<<gridDim, blockDim, threadnum/32, stream[9]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
			DifSpmmPredictSizeWarpKernel_8<32> <<< gridDim, blockDim, 0, stream[7]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==8) //256<rowLength<=512
		{
			dim3 blockDim(512,1,1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
//			DifSpmmPredictSizeWarpKernel_16<32> <<< gridDim, blockDim, 0, stream[8]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
			DifSpmmPredictSizeOverWarpKernel_1<32,16> <<<gridDim, blockDim, 8, stream[9]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==9) //512<rowLength<=1024
		{
			dim3 blockDim(256, 1, 1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeOverWarpKernel_4<32,8> <<<gridDim, blockDim, 16, stream[9]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==10) //1024<rowLength<=2048
		{
			dim3 blockDim(512, 1, 1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeOverWarpKernel_4<32,16> <<<gridDim, blockDim, 16, stream[10]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else if(i==11) //2048<rowLength<=4096
		{
			dim3 blockDim(512, 1, 1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeOverWarpKernel_8<32,16> <<<gridDim, blockDim, 16, stream[11]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
		else //if(i==12) //rowLength>4096
		{
			dim3 blockDim(512,1, 1);
			dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
			DifSpmmPredictSizeOverWarpKernel_16<32,16> <<<gridDim, blockDim, 16, stream[12]>>>(CRowStarts.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
	}

	for(int i=0; i<13; i++)
	{
		cudaStreamDestroy(stream[i]);
	}
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_1(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);
	DifMul_1<WarpSize>(c,a,B);
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_2(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);
	DifMul_2<WarpSize>(c,a,B);
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_4(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);
	DifMul_4<WarpSize>(c,a,B);
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_8(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);
	DifMul_8<WarpSize>(c,a,B);
}

template<int WarpSize, typename T>
__global__ void __cdecl DifSpmmWarpKernel_16(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);
	DifMul_16<WarpSize>(c,a,B);
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_1(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{


	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);


	__shared__ T c_val[32];
	__shared__ uint c_indices[32];

	MulOverWarp_1<WarpSize, SegmentSize>(c,a,B,c_val,c_indices);
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_2(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{


	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);


	__shared__ T c_val[32];
	__shared__ uint c_indices[32];

	MulOverWarp_2<WarpSize, SegmentSize>(c,a,B,c_val,c_indices);
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_4(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{


	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);


	__shared__ T c_val[32];
	__shared__ uint c_indices[32];

	MulOverWarp_4<WarpSize, SegmentSize>(c,a,B,c_val,c_indices);
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_8(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{
	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);


	__shared__ T c_val[32];
	__shared__ uint c_indices[32];

	MulOverWarp_8<WarpSize, SegmentSize>(c,a,B,c_val,c_indices);

}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmColumnOverWarpKernel_16(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);


	__shared__ uint c_indices[32];

	MulOverWarpColumn_16<WarpSize, SegmentSize>(c,a,B,c_indices);
}
template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmValueOverWarpKernel_16(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{

	int tid=threadIdx.y+blockIdx.x*blockDim.y;
	if(tid>=(Queue_one[position+1]-Queue_one[position]))
	{
		return; 
	}
	int r=Queue[Queue_one[position]+tid];

	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);


    __shared__ T c_val[32];
	__shared__ uint c_indices[32];

	MulOverWarpValue_16<WarpSize, SegmentSize>(c,a,B,c_val,c_indices);

}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_8_halfup(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{

    int tid=threadIdx.y+blockIdx.x*blockDim.y;
    if(tid>=(Queue_one[position+1]-Queue_one[position]))
    {
        return; 
    }
    int r=Queue[Queue_one[position]+tid];

    CSparseVector<T> c=C.GetRow(r);
    CSparseVector<T> a=A.GetRow(r);


    __shared__ T c_val[32];
    __shared__ uint c_indices[32];

    MulOverWarp_8_halfup<WarpSize, SegmentSize, 4096>(c,a,B,c_val,c_indices);

}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_8_halfdown(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{

    int tid=threadIdx.y+blockIdx.x*blockDim.y;
    if(tid>=(Queue_one[position+1]-Queue_one[position]))
    {
        return; 
    }
    int r=Queue[Queue_one[position]+tid];

    CSparseVector<T> c=C.GetRow(r);
    CSparseVector<T> a=A.GetRow(r);


    __shared__ T c_val[32];
    __shared__ uint c_indices[32];

    MulOverWarp_8_halfdown<WarpSize, SegmentSize, 4096>(c,a,B,c_val,c_indices);
}

template<int WarpSize,  int SegmentSize, typename T>
__global__ void __cdecl DifSpmmOverWarpKernel_16(CSparseMatrixCSR<T> C,CSparseMatrixCSR<T> A,CSparseMatrixCSR<T> B,CVector<uint> Queue,CVector<uint> Queue_one, int position)
{
    int tid=threadIdx.y+blockIdx.x*blockDim.y;
    if(tid>=(Queue_one[position+1]-Queue_one[position]))
    {
        return; 
    }
    int r=Queue[Queue_one[position]+tid];

    CSparseVector<T> c=C.GetRow(r);
    CSparseVector<T> a=A.GetRow(r);


    __shared__ T c_val[32];
    __shared__ uint c_indices[32];

    MulOverWarp_16<WarpSize, SegmentSize>(c,a,B,c_val,c_indices);
}

template< typename T, typename U>
void __cdecl DifSpmmWarp(SparseDeviceMatrixCSR<U> C, SparseDeviceMatrixCSR<U> A, SparseDeviceMatrixCSR<U> B, DeviceVector<uint> Queue, DeviceVector<uint> Queue_one, T* h_queue_one)
{
    int threadnum=256;

    cudaStream_t stream[13];
    for(int i=0; i<13; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    uint count;
    for(int i=0; i<13; i++)
    {
        count = h_queue_one[i+1] - h_queue_one[i];
        if(count==0)
            continue;
        if(i==0)  //0<rowLength<=2
        {
            dim3 blockDim(2,threadnum/2,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmWarpKernel_1<2> <<< gridDim, blockDim, 0, stream[0]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==1) //2<rowLength<=4
        {
            dim3 blockDim(4,threadnum/4,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmWarpKernel_1<4> <<< gridDim, blockDim, 0, stream[1]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==2) //4<rowLength<=8
        {
            dim3 blockDim(8,threadnum/8,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmWarpKernel_1<8> <<< gridDim, blockDim, 0, stream[2]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==3) //8<rowLength<=16
        {
            dim3 blockDim(16,threadnum/16,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmWarpKernel_1<16> <<< gridDim, blockDim, 0, stream[3]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==4) //16<rowLength<=32
        {
            dim3 blockDim(32,threadnum/32,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmWarpKernel_1<32> <<< gridDim, blockDim, 0, stream[4]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==5) //32<rowLength<=64
        {
            dim3 blockDim(32,threadnum/32,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmWarpKernel_2<32> <<< gridDim, blockDim, 0, stream[5]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
            //			DifSpmmOverWarpKernel_1<32, 2> <<<gridDim, blockDim, 0, stream[9]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==6) //64<rowLength<=128
        {
            dim3 blockDim(32,8,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmWarpKernel_4<32> <<< gridDim, blockDim, 0, stream[6]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
            //			DifSpmmOverWarpKernel_2<32, 2> <<<gridDim, blockDim, 0, stream[9]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==7) //128<rowLength<=256
        {
            dim3 blockDim(32,8,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            //			DifSpmmOverWarpKernel_4<32, 2> <<<gridDim, blockDim, 0, stream[9]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
            DifSpmmWarpKernel_8<32> <<< gridDim, blockDim, 0, stream[7]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==8) //256<rowLength<=512
        {
            dim3 blockDim(512,1,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmOverWarpKernel_1<32, 16> <<<gridDim, blockDim, 0, stream[9]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
            //			DifSpmmWarpKernel_16<32> <<< gridDim, blockDim, 0, stream[8]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==9) //512<rowLength<=1024
        {
            dim3 blockDim(256,2,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmOverWarpKernel_4<32, 8> <<<gridDim, blockDim, 0, stream[9]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==10) //1024<rowLength<=2048
        {
            dim3 blockDim(512,1,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmOverWarpKernel_4<32, 16> <<<gridDim, blockDim, 0, stream[10]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
        }
        else if(i==11) //2048<rowLength<=4096
        {
            dim3 blockDim(512,1,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            DifSpmmOverWarpKernel_8<32, 16> <<<gridDim, blockDim, 0, stream[11]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);

        }
        else //if(i==12) //rowLength>4096
        {
            dim3 blockDim(512,1,1);
            dim3 gridDim(DivUp(count,(uint)blockDim.y),1,1);
            //            DifSpmmOverWarpKernel_16<32, 16> <<<gridDim, blockDim, 0, stream[12]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
            DifSpmmColumnOverWarpKernel_16<32, 16> <<<gridDim, blockDim, 0, stream[12]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
            DifSpmmOverWarpKernel_8_halfup<32, 16> <<<gridDim, blockDim, 0, stream[12]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
            
            DifSpmmOverWarpKernel_8_halfdown<32,16> <<<gridDim, blockDim, 0, stream[12]>>>(C.GetC(),A.GetC(),B.GetC(),Queue.GetC(),Queue_one.GetC(),i);
		}
	}

	for(int i=0; i<13; i++)
	{
		cudaStreamDestroy(stream[i]);
	}
}

#endif