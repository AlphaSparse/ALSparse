#ifndef CSRSPGEMM_DEVICE_AMGX
#define CSRSPGEMM_DEVICE_AMGX
#include "alphasparse.h"
#include <thrust/scan.h>
#include "amgx/amgx_util.h"
#include "amgx/hash_index.h"
#include "amgx/hash_containers_sm70.h"

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

template< int NUM_THREADS_PER_ROW, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
__global__ __launch_bounds__( CTA_SIZE )
void
count_non_zeroes_kernel( const int A_num_rows,
                         const int *__restrict A_rows,
                         const int *__restrict A_cols,
                         const int *__restrict B_rows,
                         const int *__restrict B_cols,
                         int *__restrict C_rows,
                         int *__restrict C_cols,
                         int *Aq1,
                         int *Bq1,
                         int *Aq2,
                         int *Bq2,
                         const int gmem_size,
                         int *g_keys,
                         int *wk_work_queue,
                         int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // The hash keys stored in shared memory.
    __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id_u = warp_id( );
    const int lane_id_u = lane_id( );
    // Constants.
    const int lane_id_div_num_threads = lane_id_u / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id_u % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id_u;
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id_u * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );

    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id_u ) )
    {
        int c_row_id = a_row_id;

        if (Aq1 != NULL)
        {
            a_row_id = Aq1[a_row_id];
        }

        // Make sure we have to proceed.
        if ( COUNT_ONLY )
        {
            volatile int *status = reinterpret_cast<volatile int *>( wk_status );

            if ( set.has_failed() || *status != 0 )
            {
                return;
            }
        }

        // Clear the set.
        set.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id_u < 2 )
        {
            a_col_tmp = Ld<LD_NC>::load( &A_rows[a_row_id + lane_id_u] );
        }

        int a_col_it  = shfl( a_col_tmp, 0 );
        int a_col_end = shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id_u ; u_any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id = -1;

            if ( is_active )
            {
                b_row_id = Ld<LD_NC>::load( &A_cols[a_col_it] );

                //b_row_id is actually column of A
                if (Aq2 != NULL)
                {
                    b_row_id = Aq2[b_row_id];
                }

                if (Bq1 != NULL)
                {
                    b_row_id = Bq1[b_row_id];
                }
            }

            const int num_rows = __popc( u_ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_b_row_id = shfl( b_row_id, local_k );
                // Load the range of the row of B.
                int b_col_tmp = -1;

                if ( is_active_k && lane_id_mod_num_threads < 2 )
                {
                    b_col_tmp = Ld<LD_NC>::load( &B_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                }

                int b_col_it  = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                int b_col_end = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                // Iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; u_any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = Ld<LD_NC>::load( &B_cols[b_col_it] );

                        // b_col_id is actually column of B
                        if (Bq2 != NULL)
                        {
                            b_col_id = Bq2[b_col_id];
                        }
                    }

                    set.insert( b_col_id, COUNT_ONLY ? wk_status : NULL );
                }
            }
        }

        // Store the results.
        if ( COUNT_ONLY )
        {
            int count = set.compute_size();

            if ( lane_id_u == 0 )
            {
                C_rows[c_row_id] = count;
            }
        }
        else
        {
            int c_col_tmp = -1;

            if ( lane_id_u < 2 )
            {
                c_col_tmp = Ld<LD_NC>::load( &C_rows[c_row_id + lane_id_u] );
            }

            int c_col_it  = shfl( c_col_tmp, 0 );
            int c_col_end = shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            set.store( count, &C_cols[c_col_it] );
        }
    }
}

template< int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE, bool COUNT_ONLY >
__global__ __launch_bounds__( CTA_SIZE )
void
count_non_zeroes_kernel( const int A_num_rows,
                         const int *A_rows,
                         const int *A_cols,
                         const int *B_rows,
                         const int *B_cols,
                         int *C_rows,
                         int *C_cols,
                         int *Aq1,
                         int *Bq1,
                         int *Aq2,
                         int *Bq2,
                         const int gmem_size,
                         int *g_keys,
                         int *wk_work_queue,
                         int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // The hash keys stored in shared memory.
    __shared__ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id_u = warp_id();
    const int lane_id_u = lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id_u;
    // Create local storage for the set.
    Hash_set<int, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id_u * SMEM_SIZE], &g_keys[a_row_id * gmem_size], gmem_size );

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id_u ) )
    {
        int c_row_id = a_row_id;

        if (Aq1 != NULL)
        {
            a_row_id = Aq1[a_row_id];
        }

        // Make sure we have to proceed.
        if ( COUNT_ONLY )
        {
            volatile int *status = reinterpret_cast<volatile int *>( wk_status );

            if ( set.has_failed() || *status != 0 )
            {
                return;
            }
        }

        // Clear the set.
        set.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id_u < 2 )
        {
            a_col_tmp = Ld<LD_NC>::load( &A_rows[a_row_id + lane_id_u] );
        }

        int a_col_it  = shfl( a_col_tmp, 0 );
        int a_col_end = shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id_u ; u_any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id = -1;

            if ( is_active )
            {
                b_row_id = Ld<LD_NC>::load( &A_cols[a_col_it] );

                //b_row_id is actually column of A
                if (Aq2 != NULL)
                {
                    b_row_id = Aq2[b_row_id];
	    		
                }

                if (Bq1 != NULL)
                {
                    b_row_id = Bq1[b_row_id];
                }
            }

            // The number of valid rows.
            const int num_rows = __popc( u_ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_b_row_id = shfl( b_row_id, k );
                // Load the range of the row of B.
                int b_col_tmp = -1;

                if ( lane_id_u < 2 )
                {
                    b_col_tmp = Ld<LD_NC>::load( &B_rows[uniform_b_row_id + lane_id_u] );
                }

                int b_col_it  = shfl( b_col_tmp, 0 );
                int b_col_end = shfl( b_col_tmp, 1 );

                // Iterate over the range of columns of B.
                for ( b_col_it += lane_id_u ; u_any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                {
                    int b_col_id = -1;

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = Ld<LD_NC>::load( &B_cols[b_col_it] );

                        // b_col_id is actually column of B
                        if (Bq2 != NULL)
                        {
                            b_col_id = Bq2[b_col_id];
                        }
                    }

                    set.insert( b_col_id, COUNT_ONLY ? wk_status : NULL );
                }
            }
        }

        // Store the results.
        if ( COUNT_ONLY )
        {
            int count = set.compute_size();

            if ( lane_id_u == 0 )
            {
                C_rows[c_row_id] = count;
            }
        }
        else
        {
            int c_col_tmp = -1;

            if ( lane_id_u < 2 )
            {
                c_col_tmp = Ld<LD_NC>::load( &C_rows[c_row_id + lane_id_u] );
            }

            int c_col_it  = shfl( c_col_tmp, 0 );
            int c_col_end = shfl( c_col_tmp, 1 );
            // Store the results.
            int count = c_col_end - c_col_it;

            if ( count == 0 )
            {
                continue;
            }

            set.store( count, &C_cols[c_col_it] );
        }
    }
}

template< int NUM_THREADS_PER_ROW, typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE, 6 )
void
compute_values_kernel( const int A_num_rows,
                       const int *__restrict A_rows,
                       const int *__restrict A_cols,
                       const Value_type *__restrict A_vals,
                       const int *__restrict B_rows,
                       const int *__restrict B_cols,
                       const Value_type *__restrict B_vals,
                       const int *__restrict C_rows,
                       int *__restrict C_cols,
                       Value_type *__restrict C_vals,
                       Value_type alpha,
                       int *Aq1,
                       int *Bq1,
                       int *Aq2,
                       int *Bq2,
                       const int gmem_size,
                       int *g_keys,
                       Value_type *g_vals,
                       int *wk_work_queue,
                       int *wk_status )
{

    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;
    // The hash keys stored in shared memory.
    __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id_u = warp_id();
    const int lane_id_u = lane_id();
    // Constants.
    const int lane_id_div_num_threads = lane_id_u / NUM_THREADS_PER_ROW;
    const int lane_id_mod_num_threads = lane_id_u % NUM_THREADS_PER_ROW;
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id_u;
    // Create local storage for the set.
    Hash_map<int, Value_type, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id_u * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vals[warp_id_u * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id_u ) )
    {
        int c_row_id = a_row_id;

        if (Aq1 != NULL)
        {
            a_row_id = Aq1[a_row_id];
        }

        // Clear the map.
        map.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id_u < 2 )
        {
            a_col_tmp = Ld<LD_NC>::load( &A_rows[a_row_id + lane_id_u] );
        }

        int a_col_it  = shfl( a_col_tmp, 0 );
        int a_col_end = shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id_u ; u_any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id(-1);
            Value_type a_value = Value_type{};

            if ( is_active )
            {
                b_row_id = Ld<LD_NC>::load( &A_cols[a_col_it] );
                a_value  = Ld<LD_NC>::load( &A_vals[a_col_it] );

                //b_row_id is actually column of A
                if (Aq2 != NULL)
                {
                    b_row_id = Aq2[b_row_id];
                }

                if (Bq1 != NULL)
                {
                    b_row_id = Bq1[b_row_id];
                }
            }

            const int num_rows = __popc( u_ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
            {
                int local_k = k + lane_id_div_num_threads;
                // Is it an active thread.
                bool is_active_k = local_k < num_rows;
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_b_row_id = shfl( b_row_id, k + lane_id_div_num_threads );
                // The value of A.
                const Value_type uniform_a_value = shfl( a_value, k + lane_id_div_num_threads );
                // Load the range of the row of B.
                int b_col_tmp = -1;

                if ( is_active_k && lane_id_mod_num_threads < 2 )
                {
                    b_col_tmp = Ld<LD_NC>::load( &B_rows[uniform_b_row_id + lane_id_mod_num_threads] );
                }

                int b_col_it  = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 0 );
                int b_col_end = shfl( b_col_tmp, lane_id_div_num_threads * NUM_THREADS_PER_ROW + 1 );

                // Iterate over the range of columns of B.
                for ( b_col_it += lane_id_mod_num_threads ; u_any(b_col_it < b_col_end) ; b_col_it += NUM_THREADS_PER_ROW )
                {
                    int b_col_id(-1);
                    Value_type b_value = Value_type{};

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = Ld<LD_NC>::load( &B_cols[b_col_it] );
                        b_value  = Ld<LD_NC>::load( &B_vals[b_col_it] );

                        //b_col_id is actually column of B
                        if (Bq2 != NULL)
                        {
                            b_col_id = Bq2[b_col_id];
                        }
                    }

                    map.insert( b_col_id, uniform_a_value * b_value, wk_status );
                }
            }
        }

        // Store the results.
        int c_col_tmp = -1;

        if ( lane_id_u < 2 )
        {
            c_col_tmp = Ld<LD_NC>::load( &C_rows[c_row_id + lane_id_u] );
        }

        int c_col_it  = shfl( c_col_tmp, 0 );
        int c_col_end = shfl( c_col_tmp, 1 );
        // Store the results.
        int count = c_col_end - c_col_it;

        if ( count == 0 )
        {
            continue;
        }

        map.store( count, &C_cols[c_col_it], &C_vals[c_col_it] );
    }

}

template< typename Value_type, int CTA_SIZE, int SMEM_SIZE, int WARP_SIZE >
__global__ __launch_bounds__( CTA_SIZE, 6 )
void
compute_values_kernel( const int A_num_rows,
                       const int *__restrict A_rows,
                       const int *__restrict A_cols,
                       const Value_type *__restrict A_vals,
                       const int *__restrict B_rows,
                       const int *__restrict B_cols,
                       const Value_type *__restrict B_vals,
                       const int *__restrict C_rows,
                       int *__restrict C_cols,
                       Value_type *__restrict C_vals,
                       Value_type alpha,
                       int *Aq1,
                       int *Bq1,
                       int *Aq2,
                       int *Bq2,
                       const int gmem_size,
                       int *g_keys,
                       Value_type *g_vals,
                       int *wk_work_queue,
                       int *wk_status )
{
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    // The hash keys stored in shared memory.
    __shared__ /*volatile*/ int s_keys[NUM_WARPS * SMEM_SIZE];
    // The hash values stored in shared memory.
    __shared__ Value_type s_vals[NUM_WARPS * SMEM_SIZE];
    // The coordinates of the thread inside the CTA/warp.
    const int warp_id_u = warp_id();
    const int lane_id_u = lane_id();
    // First threads load the row IDs of A needed by the CTA...
    int a_row_id = blockIdx.x * NUM_WARPS + warp_id_u;
    // Create local storage for the set.
    Hash_map<int, Value_type, SMEM_SIZE, 4, WARP_SIZE> map(&s_keys[warp_id_u * SMEM_SIZE],
            &g_keys[a_row_id * gmem_size],
            &s_vals[warp_id_u * SMEM_SIZE],
            &g_vals[a_row_id * gmem_size],
            gmem_size );

    // Loop over rows of A.
    for ( ; a_row_id < A_num_rows ; a_row_id = get_work( wk_work_queue, warp_id_u ) )
    {
        int c_row_id = a_row_id;

        if (Aq1 != NULL)
        {
            a_row_id = Aq1[a_row_id];
        }

        // Clear the map.
        map.clear();
        // Load the range of the row.
        int a_col_tmp = -1;

        if ( lane_id_u < 2 )
        {
            a_col_tmp = Ld<LD_NC>::load( &A_rows[a_row_id + lane_id_u] );
        }

        int a_col_it  = shfl( a_col_tmp, 0 );
        int a_col_end = shfl( a_col_tmp, 1 );

        // Iterate over the columns of A.
        for ( a_col_it += lane_id_u ; u_any(a_col_it < a_col_end) ; a_col_it += WARP_SIZE )
        {
            // Is it an active thread.
            const bool is_active = a_col_it < a_col_end;
            // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
            int b_row_id = -1;
            Value_type a_value = Value_type{};

            if ( is_active )
            {
                b_row_id = Ld<LD_NC>::load( &A_cols[a_col_it] );
                a_value  = Ld<LD_NC>::load( &A_vals[a_col_it] );

                //b_row_id is actually column of A
                if (Aq2 != NULL)
                {
                    b_row_id = Aq2[b_row_id];
                }

                if (Bq1 != NULL)
                {
                    b_row_id = Bq1[b_row_id];
                }
            }

            const int num_rows = __popc( u_ballot(is_active) );

            // Uniform loop: threads collaborate to load other elements.
            for ( int k = 0 ; k < num_rows ; ++k )
            {
                // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
                const int uniform_b_row_id = shfl( b_row_id, k );
                // The value of A.
                const Value_type uniform_a_value = shfl( a_value, k ) * alpha;
                // Load the range of the row of B.
                int b_col_tmp = -1;

                if ( lane_id_u < 2 )
                {
                    b_col_tmp = Ld<LD_NC>::load( &B_rows[uniform_b_row_id + lane_id_u] );
                }

                int b_col_it  = shfl( b_col_tmp, 0 );
                int b_col_end = shfl( b_col_tmp, 1 );

                // Iterate over the range of columns of B.
                for ( b_col_it += lane_id_u ; u_any(b_col_it < b_col_end) ; b_col_it += WARP_SIZE )
                {
                    int b_col_id = -1;
                    Value_type b_value = Value_type{};

                    if ( b_col_it < b_col_end )
                    {
                        b_col_id = Ld<LD_NC>::load( &B_cols[b_col_it] );
                        b_value  = Ld<LD_NC>::load( &B_vals[b_col_it] );

                        if (Bq2 != NULL)
                        {
                            b_col_id = Bq2[b_col_id];
                        }
                    }

                    map.insert( b_col_id, uniform_a_value * b_value, wk_status );
                }
            }
        }

        // Store the results.
        int c_col_tmp = -1;

        if ( lane_id_u < 2 )
        {
            c_col_tmp = Ld<LD_NC>::load( &C_rows[c_row_id + lane_id_u] );
        }

        int c_col_it  = shfl( c_col_tmp, 0 );
        int c_col_end = shfl( c_col_tmp, 1 );
        // Store the results.
        int count = c_col_end - c_col_it;

        if ( count == 0 )
        {
            continue;
        }

        map.store( count, &C_cols[c_col_it], &C_vals[c_col_it] );
    }

}

template<typename T>
void count_non_zeroes(alphasparseSpMatDescr_t matA, alphasparseSpMatDescr_t matB, alphasparseSpMatDescr_t matC, int num_threads, int * m_keys, T *Aq1, T *Bq1, T *Aq2, T *Bq2 )
{
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 256;
    const int WARP_SIZE = 32;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    const int SMEM_SIZE = 128;
    int m_gmem_size = 512;
    // Reset work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    int * m_work_queue;
    cudaMalloc((void **)&m_work_queue, sizeof(int));
    cudaMemcpy( m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) ;

    int * m_status = NULL;
    cudaMalloc((void **)&m_status, sizeof(int));
    int status = 0;
    cudaMemcpy( m_status, &status, sizeof(int), cudaMemcpyHostToDevice );
    // Compute non-zero elements.
    switch ( num_threads )
    {
        case 2:
            count_non_zeroes_kernel< 2, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (int*)matC->row_data,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

        case 4:
            count_non_zeroes_kernel< 4, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (int*)matC->row_data,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

        case 8:
            count_non_zeroes_kernel< 8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (int*)matC->row_data,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

        case 16:
            count_non_zeroes_kernel<16, CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (int*)matC->row_data,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
            break;

        default:
            count_non_zeroes_kernel<CTA_SIZE, SMEM_SIZE, WARP_SIZE, true> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (int*)matC->row_data,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                m_keys,
                m_work_queue,
                m_status );
    }    
    CHECK_CUDA( cudaGetLastError() );
}

template<typename T, typename U>
int compute_values(alphasparseSpMatDescr_t matA, alphasparseSpMatDescr_t matB, alphasparseSpMatDescr_t matC, int num_threads, int * m_keys, U * m_vals, U alpha, T *Aq1, T *Bq1, T *Aq2, T *Bq2  )
{
    const int GRID_SIZE = 1024;
    const int CTA_SIZE  = 128;
    const int SMEM_SIZE = 128;
    const int WARP_SIZE = 32;
    const int NUM_WARPS = CTA_SIZE / WARP_SIZE;
    int m_gmem_size = 512;
    // Reset the work queue.
    int work_offset = GRID_SIZE * NUM_WARPS;
    int * m_work_queue;
    cudaMalloc((void **)&m_work_queue, sizeof(int));
    cudaMemcpy( m_work_queue, &work_offset, sizeof(int), cudaMemcpyHostToDevice ) ;

    // Compute the values.
    int status = 0;
    int * m_status = NULL;
    // cudaMalloc((void **)&m_status, sizeof(int));
    // cudaMemcpy( m_status, &status, sizeof(int), cudaMemcpyHostToDevice );		

    switch ( num_threads )
    {
        case 2:
            compute_values_kernel< 2, U, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (U*)matA->val_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (U*)matB->val_data,
                (int*)matC->row_data,
                (int*)matC->col_data,
                (U*)matC->val_data,
                alpha,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                (int*)m_keys,
                m_vals,
                m_work_queue,
                m_status );
            break;

        case 4:
            compute_values_kernel< 4, U, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (U*)matA->val_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (U*)matB->val_data,
                (int*)matC->row_data,
                (int*)matC->col_data,
                (U*)matC->val_data,
                alpha,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                (int*)m_keys,
                m_vals,
                m_work_queue,
                m_status );
            break;

        case 8:
            compute_values_kernel< 8, U, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (U*)matA->val_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (U*)matB->val_data,
                (int*)matC->row_data,
                (int*)matC->col_data,
                (U*)matC->val_data,
                alpha,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                (int*)m_keys,
                m_vals,
                m_work_queue,
                m_status );
            break;

        case 16:
            compute_values_kernel<16, U, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (U*)matA->val_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (U*)matB->val_data,
                (int*)matC->row_data,
                (int*)matC->col_data,
                (U*)matC->val_data,
                alpha,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                (int*)m_keys,
                m_vals,
                m_work_queue,
                m_status );
            break;

        default:
            compute_values_kernel<U, CTA_SIZE, SMEM_SIZE, WARP_SIZE> <<< GRID_SIZE, CTA_SIZE>>>(
                (int)matA->rows,
                (int*)matA->row_data,
                (int*)matA->col_data,
                (U*)matA->val_data,
                (int*)matB->row_data,
                (int*)matB->col_data,
                (U*)matB->val_data,
                (int*)matC->row_data,
                (int*)matC->col_data,
                (U*)matC->val_data,
                alpha,
                NULL,
                NULL,
                NULL,
                NULL,
                m_gmem_size,
                (int*)m_keys,
                m_vals,
                m_work_queue,
                m_status );
    }    
    CHECK_CUDA( cudaGetLastError() );
    // cudaMemcpy( &status, m_status, sizeof(int), cudaMemcpyDeviceToHost );	
    return status;
}

#endif