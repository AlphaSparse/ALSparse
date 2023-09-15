#include "alphasparse.h"
#include "csrspgemm_device_ac.h"
#include "ac/default_scheduling_traits.h"
#include "ac/Multiply.h"

template<typename IndexType, typename DataType>
__global__ void mul_alpha(DataType * val, IndexType size, const DataType a)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int str = blockDim.x * gridDim.x;
    for(int i = tid; i < size; i += str)
        val[i] *= a;
}

template <typename IndexType, typename DataType>
alphasparseStatus_t spgemm_csr_ac(alphasparseHandle_t handle,
                        alphasparseOperation_t opA,
                        alphasparseOperation_t opB,
                        const DataType alpha,
                        alphasparseSpMatDescr_t matA,
                        alphasparseSpMatDescr_t matB,
                        const DataType beta,
                        alphasparseSpMatDescr_t matC,
                        void * externalBuffer2)
{
    const int Threads = 256;
    const int BlocksPerMP = 3;
    const int NNZPerThread = 2;
    const int InputElementsPerThreads = 4;
    const int RetainElementsPerThreads = 4;
    const int MaxChunksToMerge = 16;
    const int MaxChunksGeneralizedMerge = 512; // MAX: 865
    const int MergePathOptions = 8;
    const int RealBlocksPerMP = (256 * BlocksPerMP + Threads - 1) / Threads;

    GPUMatrixMatrixMultiplyTraits DefaultTraits(Threads, BlocksPerMP, NNZPerThread, InputElementsPerThreads, RetainElementsPerThreads, MaxChunksToMerge, MaxChunksGeneralizedMerge, MergePathOptions); // DefaultTraits(128, 2, 4, 1, 8, 128, 8);
    DefaultTraits.preferLoadBalancing = true;
    ACSpGEMM::MultiplyImplementation<IndexType, DataType, 256, 3, 2, 4, 4, 16, 512, 8, false>(handle, opA, opB, alpha, matA, matB, beta, matC, (char *)externalBuffer2, DefaultTraits);
    // mul_alpha<<<1024,256>>>((DataType*)matC->val_data, matC->nnz,alpha);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
