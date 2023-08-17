#include "alphasparse.h"
#include "csrspgemm_device_fast.h"
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
#include "fast/DeviceTransfers.h"
#include "fast/SparseDeviceMatrixCSROperations.h"

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n",               \
             __LINE__,                                                         \
             cudaGetErrorString(status),                                       \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

template <typename T, typename U>
__global__ void mul_alpha(U * DATA, U a, T size)
{
    int tid = blockIdx.x * gridDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i < size; i += stride)
        DATA[i] *= a;
}

template <typename T, typename U>
alphasparseStatus_t spgemm_csr_fast(alphasparseHandle_t handle,
                        alphasparseOperation_t opA,
                        alphasparseOperation_t opB,
                        const U alpha,
                        alphasparseSpMatDescr_t matA,
                        alphasparseSpMatDescr_t matB,
                        const U beta,
                        alphasparseSpMatDescr_t matC,
                        void * externalBuffer2)
{
    std::vector<T> Queue;
    std::vector<T> Queue_one;
    double time1 = get_time_us();
    ComputeBin<T>(matA, Queue, Queue_one);
    double time2 = get_time_us();
    std::cout << "ComputeBin time: " << (time2 - time1) / (1e3) << " microseconds" << std::endl;
    float elapsed_time = (time2 - time1) / (1e3);
    // if(matC != nullptr)
    //     cudaMalloc((void **)&matC->row_data, sizeof(T)*(matA->rows + 1));

    HostVector<uint> hQueue((uint *)Queue.data(),(int64)matA->rows);
	HostVector<uint> hQueue_one((uint *)Queue_one.data(),14);

	T * h_queue_one = Queue_one.data();

    // T * DQueue, *DQueue_one;    
    // cudaMalloc((void **)&DQueue, sizeof(T) * Queue.size());
    // cudaMalloc((void **)&DQueue_one, sizeof(T)* Queue_one.size());    

    // cudaMemcpy(DQueue, Queue.data, sizeof(T) * Queue.size(), cudaMemcpyHostToDevice);
    // cudaMemcpy(DQueue_one, Queue_one.data, sizeof(T) * Queue_one.size(), cudaMemcpyHostToDevice);
    time1 = get_time_us();
    DeviceVector<uint> DQueue = ToDevice(hQueue);
	DeviceVector<uint> DQueue_one = ToDevice(hQueue_one);
    time2 = get_time_us();
    std::cout << "HTD time: " << (time2 - time1) / (1e3) << " microseconds" << std::endl;
    elapsed_time += (time2 - time1) / (1e3);
    // HostVector<U> valuesA((U *)matA->val_data, (int64)matA->nnz); 
    // HostVector<U> valuesB((U *)matB->val_data, (int64)matB->nnz); 
    // HostVector<uint> colIndicesA((uint *)matA->col_data,(int64)matA->nnz);
    // HostVector<uint> rowStartsA((uint *)matA->row_data,(int64)matA->rows);
    // HostVector<uint> colIndicesB((uint *)matB->col_data,(int64)matB->nnz);
    // HostVector<uint> rowStartsB((uint *)matB->row_data,(int64)matB->rows);

    DeviceVector<U> valuesA((int64)matA->nnz); 
    DeviceVector<U> valuesB((int64)matB->nnz); 
    DeviceVector<uint> colIndicesA((int64)matA->nnz);
    DeviceVector<uint> rowStartsA((int64)matA->rows);
    DeviceVector<uint> colIndicesB((int64)matB->nnz);
    DeviceVector<uint> rowStartsB((int64)matB->rows);

    // for(int i = 0; i < matA->rows; i++)
    //     rowStartsA[i] = (uint)matA->row_data[i];
    // for(int i = 0; i < matA->nnz; i++)
    // {
    //     valuesA[i] = ((U *)matA->val_data)[i];
    //     colIndicesA[i] = (uint)matA->col_data[i];
    // }

    // for(int i = 0; i < matB->rows; i++)
    //     rowStartsB[i] = (uint)matB->row_data[i];
    // for(int i = 0; i < matA->nnz; i++)
    // {
    //     valuesB[i] = ((U *)matB->val_data)[i];
    //     colIndicesB[i] = (uint)matB->col_data[i];
    // }

    CHECK_CUDA(cudaMemcpy(valuesA.v.Data(),matA->val_data,sizeof(U)*matA->nnz,cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(colIndicesA.v.Data(),matA->col_data,sizeof(T)*matA->nnz,cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(rowStartsA.v.Data(),matA->row_data,sizeof(T)*matA->rows,cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaMemcpy(valuesB.v.Data(),matB->val_data,sizeof(U)*matB->nnz,cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(colIndicesB.v.Data(),matB->col_data,sizeof(T)*matB->nnz,cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(rowStartsB.v.Data(),matB->row_data,sizeof(T)*matB->rows,cudaMemcpyDeviceToDevice));

    // SparseHostMatrixCSR<U> A(matA->cols, matA->rows, valuesA, colIndicesA, rowStartsA);
    // SparseHostMatrixCSR<U> B(matB->cols, matB->rows, valuesB, colIndicesB, rowStartsB);

    SparseDeviceMatrixCSR<U> A_(matA->cols, matA->rows, valuesA, colIndicesA, rowStartsA);
    SparseDeviceMatrixCSR<U> B_(matB->cols, matB->rows, valuesB, colIndicesB, rowStartsB);

    // SparseDeviceMatrixCSR<U> A_ = ToDevice(A);
	// SparseDeviceMatrixCSR<U> B_ = ToDevice(B);
    time1 = get_time_us();
    DeviceVector<uint> CrowStarts(A_.Height()+1);

	PredictCSize<T,U>(CrowStarts, A_, B_, DQueue, DQueue_one, h_queue_one);

    cudaDeviceSynchronize();
    time2 = get_time_us();
    std::cout << "PredictCSize time: " << (time2 - time1) / (1e3) << " microseconds" << std::endl;
    elapsed_time += (time2 - time1) / (1e3);
    time1 = get_time_us();
	ScanExclusive(CrowStarts);
    cudaDeviceSynchronize();
    time2 = get_time_us();
    std::cout << "ScanExclusive time: " << (time2 - time1) / (1e3) << " microseconds" << std::endl;
    elapsed_time += (time2 - time1) / (1e3);
	uint nonZeros = CrowStarts[CrowStarts.Length()-1];
	// std::cout << "Fast C nonzerod:"<<nonZeros<<std::endl;
    time1 = get_time_us();
	SparseDeviceMatrixCSR<U> C(B_.Width(), A_.Height(), DeviceVector<U>(nonZeros), DeviceVector<uint>(nonZeros), CrowStarts);
    time2 = get_time_us();
    std::cout << "SparseDeviceMatrixCSR time: " << (time2 - time1) / (1e3) << " microseconds" << std::endl;
    elapsed_time += (time2 - time1) / (1e3);
    time1 = get_time_us();
	DifSpmmWarp<T, U>(C,A_,B_,DQueue,DQueue_one, h_queue_one);
    cudaDeviceSynchronize();
    time2 = get_time_us();
    std::cout << "DifSpmmWarp time: " << (time2 - time1) / (1e3) << " microseconds" << std::endl;
    elapsed_time += (time2 - time1) / (1e3);

    std::cout << "TOTAL time: " << elapsed_time << " microseconds" << std::endl;
    CSparseMatrixCSR<U> H_C = C.GetC();
    // U * val = (U *)malloc(sizeof(U)*nonZeros);
    // CHECK_CUDA(cudaMemcpy(val,H_C.Values(),sizeof(U)*nonZeros,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(matC->row_data,H_C.RowStarts(),sizeof(T)*(A_.Height()+1),cudaMemcpyDeviceToDevice));
    // matC->row_data = (T *)C.RowStarts().Data();
    mul_alpha<<<nonZeros/256, 256>>>(H_C.Values(), alpha, nonZeros);
    if(nonZeros != matC->nnz)
    {
        matC->nnz = nonZeros;
        if(matC->col_data == nullptr)
            CHECK_CUDA(cudaMalloc((void **)&matC->col_data, sizeof(T) * nonZeros));
        if(matC->val_data == nullptr)
            CHECK_CUDA(cudaMalloc((void **)&matC->val_data, sizeof(U) * nonZeros));
        CHECK_CUDA(cudaMemcpy(matC->col_data,H_C.ColIndices(),sizeof(T)*nonZeros,cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(matC->val_data,H_C.Values(),sizeof(U)*nonZeros,cudaMemcpyDeviceToDevice));
    }    
    // matC->col_data = (T *)H_C.ColIndices();
    // matC->val_data = H_C.Values();
    

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
