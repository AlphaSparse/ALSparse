#include <cuda_runtime.h>
#include <omp.h>
#include "bitonic.h"

// #define CHECK(x) HIP_CHECK(x)

#define CHECK(x) x;

/*************   helper   *************/
__device__ int min_int(int a, int b)
{
	int temp;
	temp = (a<b) ? a : b;
	return temp;
}

/************* dense kernel *************/
#define BLOCK_SIZE1 256
#define SIZE_X 16
#define SIZE_Y 16

template<typename IndexType, typename DataType>
__global__ void matmul(const IndexType *dptr_offset_A, const IndexType* dptr_colindex_A, const DataType* dptr_value_A,
					   DataType* device_B, DataType* device_C, IndexType* dptr_offset_C, int* nnz_row_C,
					   const DataType alpha, size_t m, size_t n, size_t k)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int nnz_row = 0, offset=0;
	DataType Cvalue;
	int start_A = dptr_offset_A[row], end_A = dptr_offset_A[row+1];
	int nnz_row_A = end_A - start_A;
	int i, j, s;
	int num    = nnz_row_A / SIZE_X ;
	int remain = nnz_row_A % SIZE_X ;  
	__shared__ IndexType As_col[SIZE_X * SIZE_Y];
	__shared__ DataType As_val[SIZE_X * SIZE_Y];

	if(row < m)
	{	
		Cvalue = 0;

		for (s = 0 ; s < num; s++)
		{
			As_col[threadIdx.y * SIZE_X + threadIdx.x] = dptr_colindex_A[start_A + SIZE_X * s + threadIdx.x];
			As_val[threadIdx.y * SIZE_X + threadIdx.x] = dptr_value_A[start_A    + SIZE_X * s + threadIdx.x];

			__syncthreads();

			if (col < n)
			{
				for (i = 0; i < SIZE_X; i++)
				{
					j = As_col[threadIdx.y * SIZE_X + i];
					Cvalue += As_val[threadIdx.y * SIZE_X + i] * device_B[j * n + col];
				}
			}
			__syncthreads();
		}

		if (remain != 0) 
		{	
			if (threadIdx.x < remain)
			{
				As_col[threadIdx.y * SIZE_X + threadIdx.x] = dptr_colindex_A[start_A + SIZE_X * num + threadIdx.x];
				As_val[threadIdx.y * SIZE_X + threadIdx.x] = dptr_value_A[start_A    + SIZE_X * num + threadIdx.x];
			}

			__syncthreads();

			if (col < n)
			{
				for (i = 0; i < remain; i++)
				{
					j = As_col[threadIdx.y * SIZE_X + i];
					Cvalue += As_val[threadIdx.y * SIZE_X + i] * device_B[j * n + col];
				}
			}
		}

		if (Cvalue!=0 && col < n) 
		{
			device_C[row * n + col] = alpha * Cvalue;
			atomicAdd(&nnz_row_C[row], 1);
		}
	}
}

template<typename IndexType, typename DataType>
__global__ void trans2dense(const IndexType* dptr_offset_B, const IndexType* dptr_colindex_B, const DataType* dptr_value_B,
					   		DataType* device_B, size_t k, size_t n)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x, j;
	int colx, coly, colz, colw;
	int start = dptr_offset_B[row], end = dptr_offset_B[row+1];
	int len = 4; 
	int num = (end - start) / len;
	int remain = (end - start) % len; 
	double4 value4;
	int4 col4;

	if (row < k)
	{
		for (j = start; j < start + num * len; j+=len)
		{

			col4 = *(int4*)(dptr_colindex_B+j);
			value4 = *(double4*)(dptr_value_B+j);

			colx = col4.x;
			coly = col4.y;
			colz = col4.z;
			colw = col4.w;

			device_B[row * n + colx] = value4.x;
			device_B[row * n + coly] = value4.y;
			device_B[row * n + colz] = value4.z;
			device_B[row * n + colw] = value4.w;
		}
		if (remain > 0) 
		{
			for (j = start + num * len; j < end; j++)
			{
				colx = dptr_colindex_B[j];
				device_B[row * n + colx] = dptr_value_B[j];
			}
		}
	}
}

template<typename IndexType, typename DataType>
__global__ void trans2CSR(IndexType* dptr_offset_C, IndexType* dptr_colindex_C, DataType* dptr_value_C,
					   DataType* device_C, size_t m, size_t n)
{
	int j;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int start = dptr_offset_C[row];
	int end = dptr_offset_C[row + 1];
	int nnz = end - start;
	DataType Cvalue;
	
	int len = 4;
	int num = (n / len) * len;
	int res = n - num;

	double4 value4;

	if(row < m && nnz)  //除开不能整除的块和空行 
	{
		for(j = 0; j < num && start < end; j += len)
		{
			value4 = *(double4*)(device_C + row * n + j);
			if(value4.x)
			{
				dptr_colindex_C[start] = j;
				dptr_value_C[start] = value4.x;	
				start++;
			}
			if(value4.y)
			{
				dptr_colindex_C[start] = j + 1;
				dptr_value_C[start] = value4.y;
				start++;
			}
			if(value4.z)
			{
				dptr_colindex_C[start] = j + 2;
				dptr_value_C[start] = value4.z;
				start++;
			}
			if(value4.w)
			{
				dptr_colindex_C[start] = j + 3;
				dptr_value_C[start] = value4.w;
				start++;
			}
		}
		if(res)
		{
			for (j = num; j < n && start < end; j++)
			{
				Cvalue = device_C[row * n + j];
				if (Cvalue)
				{	
					dptr_colindex_C[start] = j;
					dptr_value_C[start] = Cvalue;
					start++;
				}
			}
		}
	} 
}


/*********** sparse kernel *************/

#define WARP_SIZE 64
#define BLOCK_SIZE 1024
#define SHARED_SIZE_SUM (BLOCK_SIZE * 2)
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 4
#define BLOCKSIZE_ESC (BLOCK_SIZE_X*BLOCK_SIZE_Y)

/***** calculate the nnz_row_C using upper method  ******/
// [1024, 1, 1]
template<typename IndexType, typename DataType>
__global__ void nnzrow_C_upper(const IndexType *dptr_offset_A, const IndexType* dptr_colindex_A,
							   const IndexType *dptr_offset_B, 
							   IndexType* nnz_row_C, size_t m)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;  
	int start_A = dptr_offset_A[row], end_A = dptr_offset_A[row+1];
	int nnz_row = 0, i, j, offset;
	int4 j_int4;
	if (row < m)
	{
		int num_ii    = (end_A - start_A) / 4;
		int remain_ii = (end_A - start_A) % 4;
		int j1, j2, j3, j4;
		for (i = 0; i < num_ii; i++)
		{
			offset = start_A + i * 4;
			j_int4 = *(int4*)(dptr_colindex_A+offset);
			j1 = j_int4.x;
			j2 = j_int4.y;
			j3 = j_int4.z;
			j4 = j_int4.w;

			nnz_row = nnz_row + dptr_offset_B[j1 + 1] - dptr_offset_B[j1]
			   + dptr_offset_B[j2 + 1] - dptr_offset_B[j2]
			   + dptr_offset_B[j3 + 1] - dptr_offset_B[j3]
			   + dptr_offset_B[j4 + 1] - dptr_offset_B[j4];
		}
		if (remain_ii)
		{
			offset = start_A + (num_ii<<2);
			for (i = 0; i < remain_ii; i++)
			{
				j = dptr_colindex_A[offset + i];
				nnz_row += dptr_offset_B[j + 1] - dptr_offset_B[j];
			}
		}
		nnz_row_C[row] = nnz_row;
	}
}

/***** calculate the offset array using prefix scan ******/
// [BLOCK_SIZE, 1, 1]
__global__ void offset_C_prefix_sum(int* input, int* output, size_t data_per_block, size_t m, int* dev_nonzeros, int* nnz_row_min, int* nnz_row_max)
{
	// data_per_block = 2 * block_size = 1024
	// input  : dptr_nnz_row_C[m]
	// output : dptr_offset_C[m+1]
	__shared__ int nnz_min;
    __shared__ int nnz_max;
	__shared__ int pre_sum;
	__shared__ int block_sum;
	__shared__ int block_data[SHARED_SIZE_SUM];
	int tid2   = threadIdx.x<<1;
	int tid2_1 = tid2 + 1;

	int num_sum = m / data_per_block;
	int remain  = m % data_per_block;

	if (threadIdx.x == 0)  
	{
		pre_sum = 0;
		nnz_min = 1<<16;
		nnz_max = -1;
	}

	for (int i = 0; i < num_sum; i++)
	{
		const size_t data_offset = i * data_per_block;
		int threadIdx2 = blockDim.x + threadIdx.x;
		block_data[threadIdx.x] = input[data_offset + threadIdx.x];
		block_data[threadIdx2]  = input[data_offset + threadIdx2];

		if (block_data[threadIdx.x] < block_data[threadIdx2])
		{
			atomicMin(&nnz_min, block_data[threadIdx.x]);
			atomicMax(&nnz_max, block_data[threadIdx2]);
		}
		else
		{
			atomicMax(&nnz_max, block_data[threadIdx.x]);
			atomicMin(&nnz_min, block_data[threadIdx2]);
		}

		//make sure all data is ready
		__syncthreads();

		//up-sweep
		//step = [0~log(data_per_block)-1]
		//offset control the step
		int offset = 1;
		for(int d = data_per_block >> 1; d > 0; d >>= 1)
		{
			//set barrier
			__syncthreads();
			if(threadIdx.x < d)
			{
				int a = tid2_1*offset - 1;
				int b = a + offset;
				block_data[b] += block_data[a];
			}
			offset <<= 1;
		}
		__syncthreads();

		//save block_sum and set it to 0
		if(threadIdx.x == 0)
		{
			block_sum = block_data[data_per_block - 1];
			block_data[data_per_block - 1] = 0;
		}

		//down_sweep
		for(int d = 1; d < data_per_block; d <<= 1)
		{
		   offset >>= 1;
		   __syncthreads();
		   if(threadIdx.x <  d)
		   {
				int a = tid2_1*offset - 1;
				int b = a + offset;
				int temp = block_data[b];
				block_data[b] += block_data[a];
				block_data[a] = temp;
		   }
		}
		__syncthreads();

		//copy shared memory to global memory
		output[data_offset + tid2]   = block_data[tid2]   + pre_sum;
		output[data_offset + tid2_1] = block_data[tid2_1] + pre_sum;

		__syncthreads();
		if (threadIdx.x == 0) pre_sum += block_sum;
	}

	if (remain > 0)
	{
		const size_t data_offset = num_sum * data_per_block;
		float f_data_per_block = (float)remain;
		int exp_data_per_block = (int) ceilf(log2f(f_data_per_block));
		data_per_block = 1<<exp_data_per_block;

		if (threadIdx.x < (data_per_block / 2))
		{
			if (threadIdx.x < (remain / 2))
			{
				block_data[tid2]   = input[data_offset + tid2];
				block_data[tid2_1] = input[data_offset + tid2_1];

				if (block_data[tid2] < block_data[tid2_1])
				{
					atomicMin(&nnz_min, block_data[tid2]);
					atomicMax(&nnz_max, block_data[tid2_1]);
				}
				else
				{
					atomicMax(&nnz_max, block_data[tid2]);
					atomicMin(&nnz_min, block_data[tid2_1]);
				}
			} 
			else
			{
				block_data[tid2]   = 0;
				block_data[tid2_1] = 0;
			}

			//make sure all data is ready
			__syncthreads();

			//up-sweep
			//step = [0~log(data_per_block)-1]
			//offset control the step
			int offset = 1;
			for(int d = data_per_block >> 1; d > 0; d >>= 1)
			{
				//set barrier
				__syncthreads();
				if(threadIdx.x < d)
				{
				  int a = tid2_1*offset - 1;
				  int b = a + offset;
				  block_data[b] += block_data[a];
				}
				offset <<= 1;
			}
			__syncthreads();

			//save block_sum and set it to 0
			if(threadIdx.x == 0)
			{
				block_sum = block_data[data_per_block - 1];
				block_data[data_per_block - 1] = 0;
			}

			//down_sweep
			for(int d = 1; d < data_per_block; d <<= 1)
			{
				offset >>= 1;
				__syncthreads();
				if(threadIdx.x <  d)
				{
				  int a = tid2_1*offset - 1;
				  int b = a + offset;
				  int temp = block_data[b];
				  block_data[b] += block_data[a];
				  block_data[a] = temp;
				}
			}
			 __syncthreads();

			//copy shared memory to global memory
			if (threadIdx.x < (remain / 2))
			{
				output[data_offset + tid2]   = block_data[tid2]   + pre_sum;
				output[data_offset + tid2_1] = block_data[tid2_1] + pre_sum;
			}
			__syncthreads();
			if (threadIdx.x == 0) pre_sum += block_sum;
		}
	}

	if (threadIdx.x == 0)
	{
		output[m] = pre_sum;
		*dev_nonzeros = pre_sum;  //nnz_C
		*nnz_row_min = nnz_min;
		*nnz_row_max = nnz_max;
	}
}

/***** bubble sort *****/
template<typename T>
__device__ void bubble_sort(int* index, T nnz)   //nnz <= 40
{
	// index: before sort ::  index[i] = i  i = 0 : nnz-1
	// index: after sort s.t. col[index[i]] <= col[index[i+1]]  i = 0 : nnz-1

	T tid2 = threadIdx.x<<1;
	T blocksize2 = blockDim.x<<1;
	
	T remain_even = (nnz / 2) % blockDim.x;
	T remain_odd = ((nnz - 1) / 2) % blockDim.x; 
	T n_i, i, offset, count;

	for (count = 0; count < nnz; count++)
	{
		if (count % 2 == 0) // step even
		{
			if (threadIdx.x < remain_even)
			{
				offset = tid2;
				if (index[offset] > index[offset + 1])
				{
					int temp = index[offset];
					index[offset] = index[offset + 1];
					index[offset + 1] = temp;
				}
			}
		}
		else  // step odd
		{
			if (threadIdx.x < remain_odd)
			{
				offset = tid2 + 1;
				if (index[offset] > index[offset + 1])
				{
					int temp = index[offset];
					index[offset] = index[offset + 1];
					index[offset + 1] = temp;
				}
			}
		}
		__syncthreads();
	}
}


/***** pre_sum the tag array *****/
template <typename T>
__device__ void tag_prefix_sum_shfl(T* input, T nnz, T* nnz_new)
{
	// input : tag
	// ouput : pre_sum
	// pre_sum: pre_sum[0] = 0
	//  			pre_sum[i] = tag[0] + ... + tag[i-1] = pre_sum[i-1] + tag[i-1]

	__shared__ T pre_sum;

	T n_i, offset;
	T a, b;
	T num_sum = nnz / WARP_SIZE;

	if (threadIdx.x == 0)  pre_sum = 0;
	__syncthreads();

	if (threadIdx.x < 64)
	{
		for (n_i = 0; n_i < num_sum; n_i++)
		{
			offset = n_i * WARP_SIZE + threadIdx.x;
			a = input[offset];

			for (ushort d = 1; d < WARP_SIZE; d <<= 1)
			{
				__syncthreads();
				b = __shfl_up(a, d, 64);

				if (threadIdx.x > (d - 1)) //
				{
					a += b;
				}
			}
			__syncthreads();

			input[offset] = a + pre_sum;

			__syncthreads();
			if(threadIdx.x == WARP_SIZE - 1)
			{
				pre_sum += a;
			}
		}
	}
	__syncthreads();

	*nnz_new = pre_sum;

}

/***** merge the col & val *****/
template<typename T, typename DataType>
__device__ void merge_val_shfl(T* pre_sum, int* col, int* col_new, 
						       DataType* val, DataType* val_new, T nnz, T nnz_new)
{
	// index: after sort s.t. col[index[i]] <= col[index[i+1]]  i = 0 : nnz-1
	// pre_sum: pre_sum[0] = 0
	//  			pre_sum[i] = tag[0] + ... + tag[i-1] = pre_sum[i-1] + tag[i-1]
	// 			pre_sum[nnz-1] = nnz_new
	// val_new[pre_sum[i+1]-1] += val[index[i]]

	T num_C, remain_C;
	T n_i, i;
	T offset;

	//set val_new to zero
	num_C    = nnz_new / blockDim.x;
	remain_C = nnz_new % blockDim.x;
	for (n_i = 0; n_i < num_C; n_i++)
	{
		i = n_i * blockDim.x + threadIdx.x;
		val_new[i] = 0.;
	}
	if (remain_C > 0)
	{
		if (threadIdx.x < remain_C)
		{
			i = num_C * blockDim.x + threadIdx.x;
			val_new[i] = 0.;
		}
	}
	__syncthreads();

	// merge
	num_C    = nnz / blockDim.x;
	remain_C = nnz % blockDim.x;
	for (n_i = 0; n_i < num_C; n_i++)
	{
		i = n_i * blockDim.x + threadIdx.x;
		offset = pre_sum[i] - 1;
		col_new[offset]  = col[i];
		atomicAdd(&val_new[offset], val[i]);
	}
	if (remain_C > 0)
	{
		if (threadIdx.x < remain_C)
		{
			i = num_C * blockDim.x + threadIdx.x;
			offset = pre_sum[i] - 1;
			col_new[offset]  = col[i];
			atomicAdd(&val_new[offset], val[i]);
		}
	}
}

/***** move the nonzeros *****/
template<typename T, typename T2, typename DataType>
__device__ void move_nonzeros_shfl(T* pre_sum, int* col, int* col_new, 
						           DataType* val, DataType* val_new, T nnz)
{
	// index: after sort s.t. col[index[i]] <= col[index[i+1]]  i = 0 : nnz-1
	// pre_sum: pre_sum[0] = 0
	//  			pre_sum[i] = tag[0] + ... + tag[i-1] = pre_sum[i-1] + tag[i-1]
	// 			pre_sum[nnz-1] = nnz_new
	// val_new[pre_sum[i+1]-1] += val[index[i]]

	T num_C, remain_C;
	T n_i, i;
	T offset, j1, j2;
	T2 j;
	bool flag = 0;

	// move
	num_C    = nnz / blockDim.x;
	remain_C = nnz % blockDim.x;
	for (n_i = 0; n_i < num_C; n_i++)
	{
		i = n_i * blockDim.x + threadIdx.x;

		if (i > 0) 
		{
			j = *(T2*)(pre_sum+i-1);
			j1 = j.x;
			j2 = j.y;
			if (j2==j1) flag = 0;
			else flag = 1;
		}
		else  //i == 0
		{
			j2 = pre_sum[i];
			if (j2) 
			{
				flag = 1;
			}
			else flag = 0;
		}

		if (flag)
		{
			offset = j2 - 1;
			col_new[offset]  = col[i];
			val_new[offset]  = val[i];
		}
	}

	if (remain_C > 0)
	{
		if (threadIdx.x < remain_C)
		{
			i = num_C * blockDim.x + threadIdx.x;


			if (i > 0) 
			{
				j = *(T2*)(pre_sum+i-1);
				j1 = j.x;
				j2 = j.y;
				if (j2==j1) flag = 0;
				else flag = 1;
			}
			else  // i == 0
			{
				j2 = pre_sum[i];
				if (j2) 
				{
					flag = 1;
				}
				else flag = 0;
			}

			if (flag)
			{
				offset = j2 - 1;
				col_new[offset]  = col[i];
				val_new[offset]  = val[i];
			}
		}
	}
}


/***** SpGEMM -- ESC ******/
// [64, 1, 1]
template <int MIN_SIZE, int MAX_SIZE, int SHARED_SIZE, int SHARED_SIZE_MAX, typename IndexType, typename DataType>
__global__ void matmul_ESC(const IndexType *dptr_offset_A, const IndexType* dptr_colindex_A, const DataType* dptr_value_A,
									const IndexType *dptr_offset_B, const IndexType* dptr_colindex_B, const DataType* dptr_value_B,
									const IndexType* dptr_offset_C, IndexType* dptr_colindex_C, DataType* dptr_value_C,
									const int* dptr_nnz_row_C_upper, int* dptr_nnz_row_C, 
									const DataType alpha, size_t m)
{
	ushort row = blockIdx.x;  // C[row, :]
	ushort n_i, n_j, i, j;
	int jj;
	int start_A = dptr_offset_A[row], end_A = dptr_offset_A[row+1];  // A[row, :]
	ushort nnz_row_A = end_A - start_A;

	int start_C;
	ushort start;
	ushort num_C, remain_C;
	ushort num_B, remain_B;
	ushort num, remain;
	bool flag_tag   = (nnz_row_A)>1?1:0;
	bool flag_merge = 0;

	float fnnz_twoupper;
	ushort nnz_twoupper;
	ushort nnz_distance;
	
	ushort nnz_row_C;
	ushort nnz_row_C_new;
	ushort nnz_row_C_new_new;

	__shared__ DataType Bs_val[SHARED_SIZE];
	__shared__ DataType Bs_val_sort[SHARED_SIZE];

	__shared__ int Bs_col[SHARED_SIZE];
	__shared__ int Cs_col[SHARED_SIZE];
	__shared__ int index[SHARED_SIZE_MAX];
	__shared__ ushort tag[SHARED_SIZE];

	int* Bs_nnz_row       = Cs_col;
	ushort* Bs_nnz_offset = tag;    // shared
	int* Bs_offset        = index;  // global

	DataType* 	As_val 	      = Bs_val_sort;


	if (row < m)
	{
		if (dptr_nnz_row_C_upper[row] <= MIN_SIZE || dptr_nnz_row_C_upper[row] > MAX_SIZE)
		{
			return;
		}

		nnz_row_C = dptr_nnz_row_C_upper[row];

		if (threadIdx.x == 0) 
		{
			dptr_nnz_row_C[row] = nnz_row_C;
		}

		if (nnz_row_C == 0) return;


		// /************ 1. read A ************/
		if (threadIdx.x < nnz_row_A)   //64
		{
			j 			        	= dptr_colindex_A[start_A + threadIdx.x];
			As_val[threadIdx.x] 	= alpha * dptr_value_A[start_A + threadIdx.x];
			Bs_offset[threadIdx.x]  = dptr_offset_B[j];
			Bs_nnz_row[threadIdx.x] = dptr_offset_B[j+1] - dptr_offset_B[j];   // B[j, :]
		}
		__syncthreads();

		// calculate the offset_B in shared_memory
		if (threadIdx.x == 0)
		{
			Bs_nnz_offset[0] = 0;
			for (i = 0; i < nnz_row_A-1; i++)
			{
				Bs_nnz_offset[i+1] = Bs_nnz_offset[i] + Bs_nnz_row[i];  // B_shared offset
			}
		}
		__syncthreads();
		

		/************ 2. read B ************/
		ushort num_A    = nnz_row_A / BLOCK_SIZE_Y;
		ushort remain_A = nnz_row_A % BLOCK_SIZE_Y;
		ushort tid_x 	= threadIdx.x % BLOCK_SIZE_X;
		ushort tid_y 	= threadIdx.x / BLOCK_SIZE_X;

		for (n_i = 0 ; n_i < num_A; n_i++)     
		{
			i  = n_i * BLOCK_SIZE_Y + tid_y;  		 //shared  -- row
			jj = Bs_offset[i];   					 //global

			num_B    = Bs_nnz_row[i] / BLOCK_SIZE_X;
			remain_B = Bs_nnz_row[i] % BLOCK_SIZE_X;
			for (n_j = 0; n_j < num_B; n_j++)
			{
				j = n_j * BLOCK_SIZE_X + tid_x;  	 //shared  -- col
				Bs_col[Bs_nnz_offset[i] + j] = dptr_colindex_B[jj + j];
				Bs_val[Bs_nnz_offset[i] + j] = As_val[i] * dptr_value_B[jj + j];
			}
			if (remain_B > 0)
			{
				if (tid_x < remain_B)
				{
					j = num_B * BLOCK_SIZE_X + tid_x;   //shared
					Bs_col[Bs_nnz_offset[i] + j] = dptr_colindex_B[jj + j];
					Bs_val[Bs_nnz_offset[i] + j] = As_val[i] * dptr_value_B[jj + j];
				}
			}
		}

		if (remain_A > 0) 
		{	
			if (tid_y < remain_A)
			{
				i  = num_A * BLOCK_SIZE_Y + tid_y;   //shared
				jj = Bs_offset[i];   				 //global

				num_B    = Bs_nnz_row[i] / BLOCK_SIZE_X;
				remain_B = Bs_nnz_row[i] % BLOCK_SIZE_X;
				for (n_j = 0; n_j < num_B; n_j++)
				{
					j = n_j * BLOCK_SIZE_X + tid_x;   //shared
					Bs_col[Bs_nnz_offset[i] + j] = dptr_colindex_B[jj + j];
					Bs_val[Bs_nnz_offset[i] + j] = As_val[i] * dptr_value_B[jj + j];
				}
				if (remain_B > 0)
				{
					if (tid_x < remain_B)
					{
						j = num_B * BLOCK_SIZE_X + tid_x;   //shared
						Bs_col[Bs_nnz_offset[i] + j] = dptr_colindex_B[jj + j];
						Bs_val[Bs_nnz_offset[i] + j] = As_val[i] * dptr_value_B[jj + j];
					}
				}
			}
		}

		/************ 3.1 bitonic sort colindex_B  ***********************/ 
		// nnz_row_C = 1
		if (nnz_row_C == 1)
		{
			if (threadIdx.x == 0)
			{
				dptr_nnz_row_C[row] = 1;
				start_C = dptr_offset_C[row];
				dptr_colindex_C[start_C] = Bs_col[0];
				dptr_value_C[start_C]    = Bs_val[0];
			}
			
			return;
		} 

		/********** 3.1.1 Code index *************/
		// index[i] = (col[i] * 4096) + i
		num_C	 = nnz_row_C / blockDim.x;
		remain_C = nnz_row_C % blockDim.x;
		for (n_i = 0; n_i < num_C; n_i++)
		{
			i = n_i * blockDim.x + threadIdx.x;
			index[i] = ((int)(Bs_col[i])<<12) + i;
		}
		if (remain_C > 0)
		{
			if (threadIdx.x < remain_C)
			{
				i = num_C * blockDim.x + threadIdx.x;
				index[i] = ((int)(Bs_col[i])<<12) + i;
			}
		}
		__syncthreads();


		/*********** 3.1.2 adative select sorting method *******/

		if (MAX_SIZE <= 128 && nnz_row_C <= 40)
		{
			if (threadIdx.x < 20)
				bubble_sort<ushort>(index, nnz_row_C);
		}
		else
		{
			/*********** 3.1.2.1 padding the sorting array *******/
			int exp_twoupper = ceilf(log2f(float(nnz_row_C)));
			int twoupper     = 1<<exp_twoupper;
			int twodistance  = twoupper - nnz_row_C;

			if (twodistance > 0)
			{
				num    = twodistance / blockDim.x;
				remain = twodistance % blockDim.x;
				for (n_i = 0; n_i < num; n_i++)
				{
					int start = nnz_row_C + n_i * blockDim.x + threadIdx.x;
					index[start] = (m<<12) + start;
				}
				__syncthreads();
				if (remain > 0)
				{
					if (threadIdx.x < remain)
					{
						int start = nnz_row_C + num * blockDim.x + threadIdx.x;
						index[start] = (m<<12) + start;
					}
				}
			}
			__syncthreads();

			if (threadIdx.x < 64)
				bitonic_sort_on_fast(index, twoupper);   //


		}
		__syncthreads();

		/************* 3.1.3  decode the index  ***************/
		for (n_i = 0; n_i < num_C; n_i++)
		{
			start = threadIdx.x + n_i * blockDim.x;
			int temp = index[start];
			int i = temp & 0xFFF;
			Bs_col[start] = (temp >> 12);
			Bs_val_sort[start] = Bs_val[i];
		}
		if (remain_C > 0)
		{
			if (threadIdx.x < remain_C)
			{
				start = threadIdx.x + num_C * blockDim.x;
				int temp = index[start];
				int i = temp & 0xFFF;
				Bs_col[start] = (temp >> 12);
				Bs_val_sort[start] = Bs_val[i];
			}
		}

		/************ 3.2 merge colindex_C and value_C ************/
		/************ 3.2.1 tag the duplicate ones ************/
		if (flag_tag)
		{
			if (threadIdx.x == 0) tag[0] = 1;

			// shfl
			ushort blocksize_shf = 63;
			num_C    = (nnz_row_C - 1) / blocksize_shf;
			remain_C = (nnz_row_C - 1) % blocksize_shf;

			if (threadIdx.x < 64)
			{
				for (n_i = 0; n_i < num_C; n_i++) 
				{
					start = threadIdx.x + n_i * blocksize_shf;

					ushort a = Bs_col[start];
					ushort b = __shfl_up(a, 1, 64);
					if( threadIdx.x > 0 )
					{
						if( a==b ) tag[start] = 0;
						else tag[start] = 1;
					}
				}
				if (remain_C > 0) 
				{
					if (threadIdx.x < remain_C + 1)
					{
						start = threadIdx.x + num_C * blocksize_shf;
						ushort a = Bs_col[start];
						ushort b = __shfl_up(a, 1, 64);
						if( threadIdx.x > 0 )
						{
							if( a==b ) tag[start] = 0;
							else tag[start] = 1;
						}
					}
				}
			}
			__syncthreads();

			/************ 3.2.2 calculate the prefix sum of tag ************/
			//presum shfl_up -- padding

			num_C   	 = nnz_row_C / (WARP_SIZE);
			remain_C     = nnz_row_C % (WARP_SIZE);
			if (remain_C)
			{
				nnz_twoupper = (num_C + 1) * (WARP_SIZE);
				nnz_distance = nnz_twoupper - nnz_row_C;  // WARP_SIZE
				if (threadIdx.x < nnz_distance)
				{
					start = nnz_row_C + threadIdx.x;
					tag[start] = 0;
				}
			}
			else
			{
				nnz_twoupper = nnz_row_C;
			}
			__syncthreads();

			if (threadIdx.x < 64)
				tag_prefix_sum_shfl<ushort>(tag, nnz_twoupper, &nnz_row_C_new);

			__syncthreads();
			
		}
		else
		{
			// if (threadIdx.x == 0) nnz_row_C_new = nnz_row_C;
			nnz_row_C_new = nnz_row_C;
		}


		/******* 3.2.3  merge the duplicate *******/
		/******* define if merge *******/
		if (nnz_row_C_new != nnz_row_C)
		{
			flag_merge = 1;
			merge_val_shfl(tag, Bs_col, Cs_col, 
						  Bs_val_sort, Bs_val, nnz_row_C, nnz_row_C_new);
			__syncthreads();

			if (nnz_row_C_new == 1)
			{
				if (threadIdx.x == 0)
				{
					dptr_nnz_row_C[row] = 1;
					start_C = dptr_offset_C[row];
					dptr_colindex_C[start_C] = Cs_col[0];
					dptr_value_C[start_C]    = Bs_val[0];
				}

				return;
			} 
		}

		/******* 3.2.4  tag the  nonzeros *******/
		num_C	 = nnz_row_C_new / blockDim.x;
		remain_C = nnz_row_C_new % blockDim.x;

		if (flag_merge)
		{
			for (n_i = 0; n_i < num_C; n_i++) 
			{
				start = threadIdx.x + n_i * blockDim.x;
				if (fabs(Bs_val[start]) > 1e-12 ) 
				{
					tag[start] = 1;
				}
				else tag[start] = 0;
			}
			if (remain_C > 0) 
			{
				if (threadIdx.x < remain_C)
				{
					start = threadIdx.x + num_C * blockDim.x;
					if (fabs(Bs_val[start]) > 1e-12 ) 
					{
						tag[start] = 1;
					}
					else tag[start] = 0;
				}
			}
		}
		else
		{
			for (n_i = 0; n_i < num_C; n_i++) 
			{
				start = threadIdx.x + n_i * blockDim.x;
				if (fabs(Bs_val_sort[start]) > 1e-12 ) 
				{
					tag[start] = 1;
				}
				else tag[start] = 0;
			}
			if (remain_C > 0) 
			{
				if (threadIdx.x < remain_C)
				{
					start = threadIdx.x + num_C * blockDim.x;
					if (fabs(Bs_val_sort[start]) > 1e-12 ) 
					{
						tag[start] = 1;
					}
					else tag[start] = 0;
				}
			}
		}
		__syncthreads();

		/******* 3.2.4  padding the tag *******/

		//presum shfl_up -- padding
		num_C   	 = nnz_row_C_new / (WARP_SIZE);
		remain_C     = nnz_row_C_new % (WARP_SIZE);
		if (remain_C)
		{
			nnz_twoupper = (num_C + 1) * (WARP_SIZE);
			nnz_distance = nnz_twoupper - nnz_row_C_new;  // WARP_SIZE
			if (threadIdx.x < nnz_distance)
			{
				start = nnz_row_C_new + threadIdx.x;
				tag[start] = 0;
			}
		}
		else
		{
			nnz_twoupper = nnz_row_C_new;
		}
		__syncthreads();

		/******* 3.2.5 prefix sum the tag *******/
		if (threadIdx.x < 64)
			tag_prefix_sum_shfl<ushort>(tag, nnz_twoupper, &nnz_row_C_new_new);

		__syncthreads();

		/************ 4. write colindex_C and value_C ************/
		if (nnz_row_C_new != nnz_row_C_new_new)
		{
			if (threadIdx.x == 0) 
			{
				dptr_nnz_row_C[row] = nnz_row_C_new_new;  // update nnz_C
			}

			if (nnz_row_C_new_new == 0)
			{
				return;				
			}

			if (flag_merge)
			{
				move_nonzeros_shfl<ushort, ushort2>(tag, Cs_col, Bs_col, 
						 			  Bs_val, Bs_val_sort, nnz_row_C_new);
				__syncthreads();

				num_C	 = nnz_row_C_new_new / blockDim.x;
				remain_C = nnz_row_C_new_new % blockDim.x;
				for (n_i = 0 ; n_i < num_C; n_i++)
				{
					i       = n_i * blockDim.x;   //shared
					start_C = dptr_offset_C[row] + i;   //global

					dptr_colindex_C[start_C + threadIdx.x] = Bs_col[i + threadIdx.x];
					dptr_value_C[start_C + threadIdx.x]    = Bs_val_sort[i + threadIdx.x];
				}

				if (remain_C > 0) 
				{	
					if (threadIdx.x < remain_C)
					{
						i       = num_C * blockDim.x;   //shared
						start_C = dptr_offset_C[row] + i;   //global

						dptr_colindex_C[start_C + threadIdx.x] = Bs_col[i + threadIdx.x];
						dptr_value_C[start_C + threadIdx.x]    = Bs_val_sort[i + threadIdx.x];
					}
				}
			}
			else
			{
				move_nonzeros_shfl<ushort, ushort2>(tag, Bs_col, Cs_col,
						 			Bs_val_sort, Bs_val, nnz_row_C_new);
				__syncthreads();

				num_C	 = nnz_row_C_new_new / blockDim.x;
				remain_C = nnz_row_C_new_new % blockDim.x;
				for (n_i = 0 ; n_i < num_C; n_i++)
				{
					i       = n_i * blockDim.x;   //shared
					start_C = dptr_offset_C[row] + i;   //global

					dptr_colindex_C[start_C + threadIdx.x] = Cs_col[i + threadIdx.x];
					dptr_value_C[start_C + threadIdx.x]    = Bs_val[i + threadIdx.x];
				}

				if (remain_C > 0) 
				{	
					if (threadIdx.x < remain_C)
					{
						i       = num_C * blockDim.x;   //shared
						start_C = dptr_offset_C[row] + i;   //global

						dptr_colindex_C[start_C + threadIdx.x] = Cs_col[i + threadIdx.x];
						dptr_value_C[start_C + threadIdx.x]    = Bs_val[i + threadIdx.x];
					}
				}
			}
		}
		else
		{
			nnz_row_C_new_new = nnz_row_C_new;

			if (threadIdx.x == 0) 
			{
				dptr_nnz_row_C[row] = nnz_row_C_new_new;  // update nnz_C
			}

			if (nnz_row_C_new_new == 0)
			{
				return;				
			}

			if (flag_merge)
			{
				num_C	 = nnz_row_C_new_new / blockDim.x;
				remain_C = nnz_row_C_new_new % blockDim.x;
				for (n_i = 0 ; n_i < num_C; n_i++)
				{
					i       = n_i * blockDim.x;   //shared
					start_C = dptr_offset_C[row] + i;   //global

					dptr_colindex_C[start_C + threadIdx.x] = Cs_col[i + threadIdx.x];
					dptr_value_C[start_C + threadIdx.x]    = Bs_val[i + threadIdx.x];
				}

				if (remain_C > 0) 
				{	
					if (threadIdx.x < remain_C)
					{
						i       = num_C * blockDim.x;   //shared
						start_C = dptr_offset_C[row] + i;   //global

						dptr_colindex_C[start_C + threadIdx.x] = Cs_col[i + threadIdx.x];
						dptr_value_C[start_C + threadIdx.x]    = Bs_val[i + threadIdx.x];
					}
				}
			}
			else
			{
				num_C	 = nnz_row_C_new_new / blockDim.x;
				remain_C = nnz_row_C_new_new % blockDim.x;
				for (n_i = 0 ; n_i < num_C; n_i++)
				{
					i       = n_i * blockDim.x;   //shared
					start_C = dptr_offset_C[row] + i;   //global

					dptr_colindex_C[start_C + threadIdx.x] = Bs_col[i + threadIdx.x];
					dptr_value_C[start_C + threadIdx.x]    = Bs_val_sort[i + threadIdx.x];
				}

				if (remain_C > 0) 
				{	
					if (threadIdx.x < remain_C)
					{
						i       = num_C * blockDim.x;   //shared
						start_C = dptr_offset_C[row] + i;   //global

						dptr_colindex_C[start_C + threadIdx.x] = Bs_col[i + threadIdx.x];
						dptr_value_C[start_C + threadIdx.x]    = Bs_val_sort[i + threadIdx.x];
					}
				}
			}
		}
	}
}


template<typename IndexType, typename DataType>
void  call_device_spgemm(alphasparseHandle_t handle,
		alphasparseOperation_t transA,
		alphasparseOperation_t transB,
		const DataType     alpha,
		const size_t       m,
		const size_t       n,
		const size_t       k,
		const size_t       nnz_A,
		const IndexType*   dptr_offset_A,
		const IndexType*   dptr_colindex_A,
		const DataType*    dptr_value_A,
		const size_t       nnz_B,
		const IndexType*   dptr_offset_B,
		const IndexType*   dptr_colindex_B,
		const DataType*    dptr_value_B,
		size_t*            ptr_nnz_C,
		IndexType*         dptr_offset_C,
		IndexType*         pdptr_colindex_C,
		DataType*          pdptr_value_C )
{
	int host_nonzeros = 1;
	int host_nonzeros_new = 1;
	// my_timer timer_kernel_malloc;   // timer

	IndexType *dptr_colindex_C = NULL;
	DataType  *dptr_value_C    = NULL;

	/*************** 0. adaptive select algorithm **************/
	if ((m >= 10000 && (nnz_B / n) >= 110) )
	{
		/************ 1. Trans device_B to dense   ************/
		DataType *device_B = NULL; 
		CHECK( cudaMalloc((void**) &device_B, (k+m)*n*sizeof(DataType)) )
		CHECK( cudaMemset(device_B, 0, (k+m)*n*sizeof(DataType)))
		DataType *device_C = device_B + k*n;

		int *dptr_nnz_row_C = NULL;
		CHECK( cudaMalloc((void**) &dptr_nnz_row_C, (m+3)*sizeof(int)) )
		CHECK( cudaMemset(dptr_nnz_row_C, 0, (m+3)*sizeof(int)))

		dim3 blocksize1(BLOCK_SIZE1, 1, 1);
		dim3 gridsize1((k-1)/BLOCK_SIZE1 + 1, 1, 1);
		trans2dense<<<gridsize1, blocksize1>>>(dptr_offset_B, dptr_colindex_B, dptr_value_B, 
										  	   device_B, k, n);

		/************  2. device_spgemm   ************/
		int *dev_nonzeros = dptr_nnz_row_C + m;
		int *nnz_row_min = dptr_nnz_row_C + m + 1;
		int *nnz_row_max = dptr_nnz_row_C + m + 2;


		dim3 blocksize_mm(SIZE_Y, SIZE_X, 1);
		dim3 gridsize_mm((m-1)/SIZE_Y+1, (n-1)/SIZE_X+1, 1);
		matmul<<<gridsize_mm, blocksize_mm>>>(dptr_offset_A, dptr_colindex_A, dptr_value_A, 
										  device_B, device_C, dptr_offset_C, dptr_nnz_row_C,
										  alpha, m, n, k);

		/************  3. Trans device_C to CSR   ************/

		// calculate nnz_c and offset_C
		dim3 blocksize_max(BLOCK_SIZE, 1, 1);
		offset_C_prefix_sum<<<1, blocksize_max>>>(dptr_nnz_row_C, dptr_offset_C, 2 * BLOCK_SIZE, m, 
												  dev_nonzeros, nnz_row_min, nnz_row_max);

		CHECK( cudaMemcpy(&host_nonzeros, dev_nonzeros, sizeof(int), cudaMemcpyDeviceToHost) )

		// Malloc pdptr_colindex_C and pdptr_value_C on Device
		CHECK( cudaMalloc((void**) &dptr_colindex_C, host_nonzeros * sizeof(IndexType)) )
		CHECK( cudaMalloc((void**) &dptr_value_C, host_nonzeros * sizeof(DataType)) )


		/************  4. Trans C to CSR on Device   ************/
		dim3 blocksize_m(BLOCK_SIZE1, 1, 1);
		dim3 gridsize_m((m-1)/BLOCK_SIZE1+1, 1, 1);
		trans2CSR<<<gridsize_m, blocksize_m>>>(dptr_offset_C, dptr_colindex_C, dptr_value_C, 
    								     device_C, m, n);    

	   /************  5. Free memory    ************/
		CHECK( cudaFree(device_B) )
		CHECK( cudaFree(dptr_nnz_row_C) )
	}
	else
	{
		/*************** 1. nnz_C_upper **************/
		int *dptr_nnz_row_C_upper = NULL; 

		CHECK(cudaMalloc((void**) &dptr_nnz_row_C_upper, (2 * m + 4)*sizeof(int)) )
		CHECK(cudaMemset(dptr_nnz_row_C_upper, 0, (2 * m + 4) * sizeof(int)))
		int *dptr_nnz_row_C = dptr_nnz_row_C_upper + m;
		int *dev_nonzeros   = dptr_nnz_row_C + m + 1;
		int *nnz_row_min    = dptr_nnz_row_C + m + 2;
		int *nnz_row_max    = dptr_nnz_row_C + m + 3;
		// 1.1  kernel -- nnz_C_upper
		dim3 blocksize_row_m(BLOCK_SIZE, 1, 1); 
		dim3 gridsize_row_m((m-1)/BLOCK_SIZE + 1, 1, 1);
		nnzrow_C_upper<<<gridsize_row_m, blocksize_row_m>>>(dptr_offset_A, dptr_colindex_A,
															dptr_offset_B, dptr_nnz_row_C_upper, m); 

		// 1.2  offset_C and nnz_C_upper
		dim3 blocksize_max(BLOCK_SIZE, 1, 1);
		offset_C_prefix_sum<<<1, blocksize_max>>>(dptr_nnz_row_C_upper, dptr_offset_C, 2 * BLOCK_SIZE, m, 
												  dev_nonzeros, nnz_row_min, nnz_row_max);
		/*************** 2. Malloc C **************/
		int host_nnz[3];
		CHECK( cudaMemcpy(host_nnz, dev_nonzeros, 3 * sizeof(int), cudaMemcpyDeviceToHost) )
		host_nonzeros = host_nnz[0];
		int host_nnz_min = host_nnz[1];
		int host_nnz_max = host_nnz[2];

		CHECK( cudaMalloc((void**) &dptr_colindex_C, host_nonzeros * sizeof(IndexType) ))
		CHECK( cudaMalloc((void**) &dptr_value_C, host_nonzeros * sizeof(DataType)) )
		CHECK(cudaMemset(dptr_colindex_C, 0, host_nonzeros * sizeof(int)))
		CHECK(cudaMemset(dptr_value_C, 0, host_nonzeros * sizeof(DataType)))
		/*************** 3. SpGEMM **************/
		dim3 blocksize_warp(BLOCKSIZE_ESC, 1, 1);
		dim3 gridsize_warp(m, 1, 1);  //m


		if (host_nnz_max > 1024)
		{
			matmul_ESC<1024, 1536, 1536, 2048><<<gridsize_warp, blocksize_warp>>>(dptr_offset_A, dptr_colindex_A, dptr_value_A,
													  			     dptr_offset_B, dptr_colindex_B, dptr_value_B,
													 				 dptr_offset_C, dptr_colindex_C, dptr_value_C,
													 				 dptr_nnz_row_C_upper, dptr_nnz_row_C, 
													 				 alpha, m);
		}

		if (host_nnz_max > 512)
		{
			matmul_ESC<512, 1024, 1024, 1024><<<gridsize_warp, blocksize_warp>>>(dptr_offset_A, dptr_colindex_A, dptr_value_A,
													  			     dptr_offset_B, dptr_colindex_B, dptr_value_B,
													 				 dptr_offset_C, dptr_colindex_C, dptr_value_C,
													 				 dptr_nnz_row_C_upper, dptr_nnz_row_C, 
													 				 alpha, m);
		}

		if (host_nnz_max > 256 && host_nnz_min < 512)
		{
			matmul_ESC<256, 512, 512, 512><<<gridsize_warp, blocksize_warp>>>(dptr_offset_A, dptr_colindex_A, dptr_value_A,
													  			     dptr_offset_B, dptr_colindex_B, dptr_value_B,
													 				 dptr_offset_C, dptr_colindex_C, dptr_value_C,
													 				 dptr_nnz_row_C_upper, dptr_nnz_row_C, 
													 				 alpha, m);
		}

		if (host_nnz_max > 128 && host_nnz_min < 256)
		{
			matmul_ESC<128, 256, 256, 256><<<gridsize_warp, blocksize_warp>>>(dptr_offset_A, dptr_colindex_A, dptr_value_A,
													  			     dptr_offset_B, dptr_colindex_B, dptr_value_B,
													 				 dptr_offset_C, dptr_colindex_C, dptr_value_C,
													 				 dptr_nnz_row_C_upper, dptr_nnz_row_C, 
													 				 alpha, m);
		}

		if (host_nnz_min < 128)
		{
			matmul_ESC<-1, 128, 128, 128><<<gridsize_warp, blocksize_warp>>>(dptr_offset_A, dptr_colindex_A, dptr_value_A,
													  			     dptr_offset_B, dptr_colindex_B, dptr_value_B,
													 				 dptr_offset_C, dptr_colindex_C, dptr_value_C,
													 				 dptr_nnz_row_C_upper, dptr_nnz_row_C, 
													 				 alpha, m);
		}




		/************   4. Free memory    ************/

		CHECK( cudaFree(dptr_nnz_row_C_upper) )
	}

	/************     Return       ************/
	*ptr_nnz_C        = host_nonzeros;
	pdptr_colindex_C = dptr_colindex_C;
	pdptr_value_C    = dptr_value_C;

}


