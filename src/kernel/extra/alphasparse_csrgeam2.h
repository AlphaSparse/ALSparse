#include "alphasparse.h"

// __global__ static void
// geam_nnz(int m,
//         int n,
//         const int *csrSortedRowPtrA,
//         const int *csrSortedColIndA,
//         const int *csrSortedRowPtrB,
//         const int *csrSortedColIndB,
//         int *row_nnz)
// {
//     // todo: can't find bugs
//     // Lane id
//     int lid = hipThreadIdx_x & (WFSIZE - 1);

//     // Wavefront id
//     int wid = hipThreadIdx_x / WFSIZE;

//     // Each wavefront processes a row
//     int row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

//     // Do not run out of bounds
//     if (row >= m) {
//         return;
//     }

//     // Row nnz marker
//     __shared__ bool stable[BLOCKSIZE];
//     bool *table = &stable[wid * WFSIZE];

//     // Get row entry and exit point of A
//     int row_begin_A = csrSortedRowPtrA[row];
//     int row_end_A   = csrSortedRowPtrA[row + 1];

//     // Get row entry and exit point of B
//     int row_begin_B = csrSortedRowPtrB[row];
//     int row_end_B   = csrSortedRowPtrB[row + 1];

//     // Load the first column of the current row from A and B to set the starting
//     // point for the first chunk
//     int col_A = (row_begin_A < row_end_A) ? csrSortedColIndA[row_begin_A] : n;
//     int col_B = (row_begin_B < row_end_B) ? csrSortedColIndB[row_begin_B] : n;

//     // Begin of the current row chunk
//     int chunk_begin = min(col_A, col_B);

//     // Initialize the row nnz for the full (wavefront-wide) row
//     int nnz = 0;

//     // Initialize the index for column access into A and B
//     row_begin_A += lid;
//     row_begin_B += lid;

//     // Loop over the chunks until the end of both rows (A and B) has been reached (which
//     // is the number of total columns n)
//     while (true) {
//         // Initialize row nnz table
//         table[lid] = false;

//         __threadfence_block();

//         // Initialize the beginning of the next chunk
//         int min_col = n;

//         // Loop over all columns of A, starting with the first entry that did not fit
//         // into the previous chunk
//         for (; row_begin_A < row_end_A; row_begin_A += WFSIZE) {
//             // Get the column of A
//             int col_A = csrSortedColIndA[row_begin_A];

//             // Get the column of A shifted by the chunk_begin
//             int shf_A = col_A - chunk_begin;

//             // Check if this column of A is within the chunk
//             if (shf_A < WFSIZE) {
//                 // Mark this column in shared memory
//                 table[shf_A] = true;
//             } else {
//                 // Store the first column index of A that exceeds the current chunk
//                 min_col = min(min_col, col_A);
//                 break;
//             }
//         }

//         // Loop over all columns of B, starting with the first entry that did not fit
//         // into the previous chunk
//         for (; row_begin_B < row_end_B; row_begin_B += WFSIZE) {
//             // Get the column of B
//             int col_B = csrSortedColIndB[row_begin_B];

//             // Get the column of B shifted by the chunk_begin
//             int shf_B = col_B - chunk_begin;

//             // Check if this column of B is within the chunk
//             if (shf_B < WFSIZE) {
//                 // Mark this column in shared memory
//                 table[shf_B] = true;
//             } else {
//                 // Store the first column index of B that exceeds the current chunk
//                 min_col = min(min_col, col_B);
//                 break;
//             }
//         }

//         __threadfence_block();

//         // Compute the chunk's number of non-zeros of the row and add it to the global
//         // row nnz counter
//         nnz += __popcll(__ballot(table[lid]));

//         // Gather wavefront-wide minimum for the next chunks starting column index
//         // Using shfl_xor here so that each thread in the wavefront obtains the final
//         // result
//         for (unsigned int i = WFSIZE >> 1; i > 0; i >>= 1) {
//             min_col = min(min_col, __shfl_xor(min_col, i));
//         }

//         // Each thread sets the new chunk beginning
//         chunk_begin = min_col;

//         // Once the chunk beginning has reached the total number of columns n,
//         // we are done
//         if (chunk_begin >= n) {
//             break;
//         }
//     }

//     // Last thread in each wavefront writes the accumulated total row nnz to global
//     // memory
//     if (lid == WFSIZE - 1) {
//         row_nnz[row] = nnz;
//     }
// }

__global__ static void
geam_nnz_per_row(int m,
                 int n,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 const int *csrSortedRowPtrB,
                 const int *csrSortedColIndB,
                 int *row_nnz)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    row_nnz[0]       = 0;
    for (int r = tid; r < n; r += stride) {
        row_nnz[r + 1] = 0;

        int as = csrSortedRowPtrA[r];
        int ae = csrSortedRowPtrA[r + 1];
        int bs = csrSortedRowPtrB[r];
        int be = csrSortedRowPtrB[r + 1];

        int ai = as, bi = bs;
        while (ai < ae && bi < be) {
            int ac = csrSortedColIndA[ai];
            int bc = csrSortedColIndB[bi];
            if (ac < bc) {
                ai++;
            } else if (ac > bc) {
                bi++;
            } else {
                ai++;
                bi++;
            }
            row_nnz[r + 1]++;
        }
        if (ai == ae) {
            row_nnz[r + 1] += be - bi;
        } else {
            row_nnz[r + 1] += ae - ai;
        }
    }
}

__global__ static void
prefix(int *row_nnz, int size)
{
    // todo: opt
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
        for (int i = 1; i < size; i++) {
            row_nnz[i] += row_nnz[i - 1];
        }
}

template<typename T>
__global__ static void
add_plain(int m,
          const T alpha,
          const T *csrSortedValA,
          const int *csrSortedRowPtrA,
          const int *csrSortedColIndA,
          const T beta,
          const T *csrSortedValB,
          const int *csrSortedRowPtrB,
          const int *csrSortedColIndB,
          T *csrSortedValC,
          const int *csrSortedRowPtrC,
          int *csrSortedColIndC,
          T* buffer)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int r = tid; r < m; r += stride) {
        int ai = csrSortedRowPtrA[r];
        int ae = csrSortedRowPtrA[r + 1];
        int bi = csrSortedRowPtrB[r];
        int be = csrSortedRowPtrB[r + 1];

        int ci = csrSortedRowPtrC[r];

        while (ai < ae && bi < be) {
            int ac = csrSortedColIndA[ai];
            int bc = csrSortedColIndB[bi];
            if (ac < bc) {
                csrSortedColIndC[ci] = ac;
                csrSortedValC[ci] = alpha * csrSortedValA[ai];
                ai++;
            } else if (ac > bc) {
                csrSortedColIndC[ci] = bc;
                csrSortedValC[ci] = beta * csrSortedValB[bi];
                bi++;
            } else {
                csrSortedColIndC[ci] = bc;
                csrSortedValC[ci] = alpha * csrSortedValA[ai] + beta * csrSortedValB[bi];
                ai++;
                bi++;
            }
            ci++;
        }
        if (ai == ae) {
            for (; bi < be; bi++, ci++) {
                csrSortedColIndC[ci] = csrSortedColIndB[bi];
                csrSortedValC[ci] = beta * csrSortedValB[bi];
            }
        } else {
            for (; ai < ae; ai++, ci++) {
                csrSortedColIndC[ci] = csrSortedColIndA[ai];
                csrSortedValC[ci] = alpha * csrSortedValA[ai];
            }
        }
    }
}