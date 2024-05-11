#include "alphasparse.h"
#include <iostream>
#include <cuComplex.h>

// __device__ __forceinline__ void print_data(const float& z)
// {
//     printf("data %f\n", z);
// }

// __device__ __forceinline__ void print_data(const double& z)
// {
//     printf("data %f\n", z);
// }

// __device__ __forceinline__ void print_data(const cuFloatComplex& z)
// {
//     printf("data real %f imag %f\n", z.x, z.y);
// }

// __device__ __forceinline__ void print_data(const cuDoubleComplex& z)
// {
//     printf("data real %f imag %f\n", z.x, z.y);
// }

// __device__ __forceinline__ float gpsv_abs(const float& z)
// {
//     return fabsf(z);
// }

// __device__ __forceinline__ double gpsv_abs(const double& z)
// {
//     return fabs(z);
// }

// __device__ __forceinline__ cuFloatComplex gpsv_abs(const cuFloatComplex& z)
// {
//     return make_cuFloatComplex(cuCabsf(z), 0.0f);
// }

// __device__ __forceinline__ cuDoubleComplex gpsv_abs(const cuDoubleComplex& z)
// {
//     return make_cuDoubleComplex(cuCabs(z), 0.0);
// }

// __device__ __forceinline__ float gpsv_sqrt(const float& z)
// {
//     return sqrtf(z);
// }

// __device__ __forceinline__ double gpsv_sqrt(const double& z)
// {
//     return sqrt(z);
// }

// __device__ __forceinline__ cuFloatComplex gpsv_sqrt(const cuFloatComplex& z)
// {
//     float x = z.x;
//     float y = z.y;

//     float sgnp = (y < 0.0f) ? -1.0f : 1.0f;
//     float absz = cuCabsf(z);

//     return make_cuFloatComplex(sqrtf((absz + x) * 0.5f), sgnp * sqrtf((absz - x) * 0.5f));
// }

// __device__ __forceinline__ cuDoubleComplex gpsv_sqrt(const cuDoubleComplex& z)
// {
//     double x = z.x;
//     double y = z.y;

//     double sgnp = (y < 0.0f) ? -1.0 : 1.0;
//     double absz = cuCabs(z);

//     return make_cuDoubleComplex(sqrt((absz + x) * 0.5), sgnp * sqrt((absz - x) * 0.5));
// }

template <unsigned int BLOCKSIZE, typename T>
__global__ static  void gpsv_strided_gather(int m,
                                            int batch_count,
                                            int batch_stride,
                                            const T* __restrict__ in,
                                            T* __restrict__ out)
{
    // Current batch this thread works on
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not run out of bounds
    if(b >= batch_count)
    {
        return;
    }

    // Process all entries of the batch
    for(int i = 0; i < m; ++i)
    {
        out[batch_count * i + b] = in[batch_stride * i + b];
    }
}

template <unsigned int BLOCKSIZE, typename T>
__global__ static void 
gpsv_interleaved_batch_householder_qr_kernel(int m,
                                            int batch_count,
                                            int batch_stride,
                                            T* __restrict__ ds,
                                            T* __restrict__ dl,
                                            T* __restrict__ d,
                                            T* __restrict__ du,
                                            T* __restrict__ dw,
                                            T* __restrict__ X,
                                            T* __restrict__ t1,
                                            T* __restrict__ t2,
                                            T* __restrict__ B)
{
    // Current batch this thread works on
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not run out of bounds
    if(b >= batch_count)
    {
        return;
    }

    T zero = {};
    // Process all rows but the last four
    for(int row = 0; row < m - 1; ++row)
    {
        bool second_last_row = (row == m - 2);

        // Some indices
        int idp0  = batch_stride * (row + 0) + b;
        int idp1  = batch_stride * (row + 1) + b;
        int idp2  = batch_stride * (row + 2) + b;
        int idp0c = batch_count * (row + 0) + b;
        int idp1c = batch_count * (row + 1) + b;
        int idp2c = batch_count * (row + 2) + b;

        // Some values
        T dlp1 = dl[idp1];
        T dp1  = d[idp1];
        T dup1 = du[idp1];
        T dwp1 = dw[idp1];
        T Bp1  = B[idp1c];

        // Prefetch
        T dsp2 = {};
        T dlp2 = {};
        T dp2  = {};
        T dup2 = {};
        T dwp2 = {};
        T Bp2  = {};

        if(!second_last_row)
        {
            dsp2 = ds[idp2];
            dlp2 = dl[idp2];
            dp2  = d[idp2];
            dup2 = du[idp2];
            dwp2 = dw[idp2];
            Bp2  = B[idp2c];
        }

        T v1 = dlp1;
        T v2 = dsp2;

        T v1v2sq = v1 * v1 + v2 * v2;
        T one = {1.0f};
        T two = {2.0f};

        if(!is_zero(v1v2sq))
        {
            T diag = d[idp0];
            T val  = alpha_sqrt(diag * diag + v1v2sq);
            T a_ii = (alpha_gt(diag, zero)) ? diag + val : diag - val;

            v1 = v1 / a_ii;
            v2 = v2 / a_ii;

            T sq   = a_ii * a_ii;
            T beta = alpha_cast<T>(two) * sq / (v1v2sq + sq);            
            T tau  = alpha_cast<T>(two)
                    / (v2 * v2 + (v1 * v1 + alpha_cast<T>(one)));

            // Process the five diagonals
            T d1 = (v2 * dsp2 + (v1 * dlp1 + diag)) * beta;
            T d2 = (v2 * dlp2 + (v1 * dp1 + du[idp0])) * beta;
            T d3 = (v2 * dp2 + (v1 * dup1 + dw[idp0])) * beta;
            T d4 = (v2 * dup2 + (v1 * dwp1 + t1[idp0c])) * beta;
            T d5 = (v2 * dwp2 + (v1 * t1[idp1c] + t2[idp0c])) * beta;
            T fs = (v2 * Bp2 + (v1 * Bp1 + B[idp0c])) * tau;

            // Update
            d[idp0] -= d1;
            du[idp0] -= d2;
            dw[idp0] -= d3;
            t1[idp0c] -= d4;
            t2[idp0c] -= d5;
            B[idp0c] -= fs;

            dl[idp1]  = v1;
            d[idp1]   = (zero - d2) * v1 + dp1;
            du[idp1]  = (zero - d3) * v1 + dup1;
            dw[idp1]  = (zero - d4) * v1 + dwp1;
            t1[idp1c] = (zero - d5) * v1 + t1[idp1c];
            B[idp1c]  = (zero - v1) * fs + Bp1;

            if(!second_last_row)
            {
                ds[idp2] = v2;
                dl[idp2] = (zero - d2) * v2 + dlp2;
                d[idp2]  = (zero - d3) * v2 + dp2;
                du[idp2] = (zero - d4) * v2 + dup2;
                dw[idp2] = (zero - d5) * v2 + dwp2;
                B[idp2c] = (zero - v2) * fs + Bp2;
            }
        }
    }

    // Backsolve
    for(int row = m - 1; row >= 0; --row)
    {
        int idp0  = batch_stride * row + b;
        int idp0c = batch_count * row + b;

        T sum = {};

        if(row + 1 < m)
            sum += du[idp0] * X[batch_stride * (row + 1) + b];
        if(row + 2 < m)
            sum += dw[idp0] * X[batch_stride * (row + 2) + b];
        if(row + 3 < m)
            sum += t1[idp0c] * X[batch_stride * (row + 3) + b];
        if(row + 4 < m)
            sum += t2[idp0c] * X[batch_stride * (row + 4) + b];

        X[idp0] = (B[idp0c] - sum) / d[idp0];
    }
}

template <unsigned int BLOCKSIZE, typename T>
__global__ static void 
gpsv_interleaved_batch_givens_qr_kernel(int m,
                                        int batch_count,
                                        int batch_stride,
                                        T* __restrict__ ds,
                                        T* __restrict__ dl,
                                        T* __restrict__ d,
                                        T* __restrict__ du,
                                        T* __restrict__ dw,
                                        T* __restrict__ r3,
                                        T* __restrict__ r4,
                                        T* __restrict__ x)
{
    int gid = threadIdx.x + BLOCKSIZE * blockIdx.x;

    if(gid >= batch_count)
    {
        return;
    }

    T zero = {};

    for(int i = 0; i < m - 2; i++)
    {
        int ind_k   = batch_stride * i + gid;
        int ind_k_1 = batch_stride * (i + 1) + gid;
        int ind_k_2 = batch_stride * (i + 2) + gid;

        // For penta diagonal matrices, need to apply two givens rotations to remove lower and lower - 1 entries
        T radius    = {};
        T cos_theta = {};
        T sin_theta = {};

        // Apply first Givens rotation
        // | cos  sin | |lk_1 dk_1 uk_1 wk_1 0   |
        // |-sin  cos | |sk_2 lk_2 dk_2 uk_2 wk_2|
        T sk_2 = ds[ind_k_2];
        T lk_1 = dl[ind_k_1];
        T lk_2 = dl[ind_k_2];
        T dk_1 = d[ind_k_1];
        T dk_2 = d[ind_k_2];
        T uk_1 = du[ind_k_1];
        T uk_2 = du[ind_k_2];
        T wk_1 = dw[ind_k_1];
        T wk_2 = dw[ind_k_2];

        radius = make_value<T>(alpha_sqrt(alpha_abs(lk_1 * conj(lk_1) + sk_2 * conj(sk_2))));

        cos_theta = conj(lk_1) / radius;
        sin_theta = conj(sk_2) / radius;

        T dlk_1_new = lk_1 * cos_theta + sk_2 * sin_theta;
        T dk_1_new  = dk_1 * cos_theta + lk_2 * sin_theta;
        T duk_1_new = uk_1 * cos_theta + dk_2 * sin_theta;
        T dwk_1_new = wk_1 * cos_theta + uk_2 * sin_theta;

        dl[ind_k_1] = dlk_1_new;
        dl[ind_k_2]
            = (zero - dk_1) * conj(sin_theta) + lk_2 * conj(cos_theta);
        d[ind_k_1] = dk_1_new;
        d[ind_k_2]
            = (zero - uk_1) * conj(sin_theta) + dk_2 * conj(cos_theta);
        du[ind_k_1] = duk_1_new;
        du[ind_k_2]
            = (zero - wk_1) * conj(sin_theta) + uk_2 * conj(cos_theta);
        dw[ind_k_1]                     = dwk_1_new;
        dw[ind_k_2]                     = wk_2 * conj(cos_theta);
        r3[batch_count * (i + 1) + gid] = wk_2 * sin_theta; 

        // Apply first Givens rotation to rhs vector
        // | cos  sin | |xk_1|
        // |-sin  cos | |xk_2|
        T xk_1     = x[ind_k_1];
        T xk_2     = x[ind_k_2];
        x[ind_k_1] = xk_1 * cos_theta + xk_2 * sin_theta;
        x[ind_k_2]
            = (zero - xk_1) * conj(sin_theta) + xk_2 * conj(cos_theta);

        // Apply second Givens rotation
        // | cos  sin | |dk   uk   wk   rk   0   |
        // |-sin  cos | |lk_1 dk_1 uk_1 wk_1 rk_1|
        lk_1   = dlk_1_new;
        T dk   = d[ind_k];
        dk_1   = dk_1_new;
        T uk   = du[ind_k];
        uk_1   = duk_1_new;
        T wk   = dw[ind_k];
        wk_1   = dwk_1_new;
        T rk   = r3[batch_count * i + gid];
        T rk_1 = r3[batch_count * (i + 1) + gid];

        radius = make_value<T>(alpha_sqrt(alpha_abs(dk * conj(dk) + lk_1 * conj(lk_1))));
        cos_theta = conj(dk) / radius;
        sin_theta = conj(lk_1) / radius;

        d[ind_k] = dk * cos_theta + lk_1 * sin_theta;
        d[ind_k_1]
            = (zero - uk) * conj(sin_theta) + dk_1 * conj(cos_theta);
        du[ind_k] = uk * cos_theta + dk_1 * sin_theta;
        du[ind_k_1]
            = (zero - wk) * conj(sin_theta) + uk_1 * conj(cos_theta);
        dw[ind_k] = wk * cos_theta + uk_1 * sin_theta;
        dw[ind_k_1]
            = (zero - rk) * conj(sin_theta) + wk_1 * conj(cos_theta);
        r3[batch_count * i + gid]       = rk * cos_theta + wk_1 * sin_theta;
        r3[batch_count * (i + 1) + gid] = rk_1 * conj(cos_theta);
        r4[batch_count * i + gid]       = rk_1 * sin_theta;

        // Apply second Givens rotation to rhs vector
        // | cos  sin | |xk  |
        // |-sin  cos | |xk_1|
        T xk     = x[ind_k];
        xk_1     = x[ind_k_1];
        x[ind_k] = xk * cos_theta + xk_1 * sin_theta;
        x[ind_k_1]
            = (zero - xk) * conj(sin_theta) + xk_1 * conj(cos_theta);
    }

    // Apply last Givens rotation
    // | cos  sin | |dk   uk   wk   rk   0   |
    // |-sin  cos | |lk_1 dk_1 uk_1 wk_1 rk_1|
    T lk_1 = dl[batch_stride * (m - 1) + gid];
    T dk   = d[batch_stride * (m - 2) + gid];
    T dk_1 = d[batch_stride * (m - 1) + gid];
    T uk   = du[batch_stride * (m - 2) + gid];
    T uk_1 = du[batch_stride * (m - 1) + gid];
    T wk   = dw[batch_stride * (m - 2) + gid];
    T wk_1 = dw[batch_stride * (m - 1) + gid];
    T rk   = r3[batch_count * (m - 2) + gid];
    T rk_1 = r3[batch_count * (m - 1) + gid];

    T radius = make_value<T>(alpha_sqrt(alpha_abs(dk * conj(dk) + lk_1 * conj(lk_1))));
    T cos_theta = conj(dk) / radius;
    T sin_theta = conj(lk_1) / radius;

    d[batch_stride * (m - 2) + gid] = dk * cos_theta + lk_1 * sin_theta;
    d[batch_stride * (m - 1) + gid]
        = (zero - uk) * conj(sin_theta) + dk_1 * conj(cos_theta);
    du[batch_stride * (m - 2) + gid] = uk * cos_theta + dk_1 * sin_theta;
    du[batch_stride * (m - 1) + gid]
        = (zero - wk) * conj(sin_theta) + uk_1 * conj(cos_theta);
    dw[batch_stride * (m - 2) + gid] = wk * cos_theta + uk_1 * sin_theta;
    dw[batch_stride * (m - 1) + gid]
        = (zero - rk) * conj(sin_theta) + wk_1 * conj(cos_theta);
    r3[batch_count * (m - 2) + gid] = rk * cos_theta + wk_1 * sin_theta;
    r3[batch_count * (m - 1) + gid] = rk_1 * conj(cos_theta);
    r4[batch_count * (m - 2) + gid] = rk_1 * sin_theta;

    // Apply last Givens rotation to rhs vector
    // | cos  sin | |xk  |
    // |-sin  cos | |xk_1|
    T xk                            = x[batch_stride * (m - 2) + gid];
    T xk_1                          = x[batch_stride * (m - 1) + gid];
    x[batch_stride * (m - 2) + gid] = xk * cos_theta + xk_1 * sin_theta;
    x[batch_stride * (m - 1) + gid]
        = (zero - xk) * conj(sin_theta) + xk_1 * conj(cos_theta);

    // Backward substitution on upper triangular R * x = x
    x[batch_stride * (m - 1) + gid]
        = x[batch_stride * (m - 1) + gid] / d[batch_stride * (m - 1) + gid];
    x[batch_stride * (m - 2) + gid]
        = (x[batch_stride * (m - 2) + gid]
           - du[batch_stride * (m - 2) + gid] * x[batch_stride * (m - 1) + gid])
          / d[batch_stride * (m - 2) + gid];

    x[batch_stride * (m - 3) + gid]
        = (x[batch_stride * (m - 3) + gid]
           - du[batch_stride * (m - 3) + gid] * x[batch_stride * (m - 2) + gid]
           - dw[batch_stride * (m - 3) + gid] * x[batch_stride * (m - 1) + gid])
          / d[batch_stride * (m - 3) + gid];

    x[batch_stride * (m - 4) + gid]
        = (x[batch_stride * (m - 4) + gid]
           - du[batch_stride * (m - 4) + gid] * x[batch_stride * (m - 3) + gid]
           - dw[batch_stride * (m - 4) + gid] * x[batch_stride * (m - 2) + gid]
           - r3[batch_count * (m - 4) + gid] * x[batch_stride * (m - 1) + gid])
          / d[batch_stride * (m - 4) + gid];

    for(int i = m - 5; i >= 0; i--)
    {
        x[batch_stride * i + gid] = (x[batch_stride * i + gid]
                                     - du[batch_stride * i + gid] * x[batch_stride * (i + 1) + gid]
                                     - dw[batch_stride * i + gid] * x[batch_stride * (i + 2) + gid]
                                     - r3[batch_count * i + gid] * x[batch_stride * (i + 3) + gid]
                                     - r4[batch_count * i + gid] * x[batch_stride * (i + 4) + gid])
                                    / d[batch_stride * i + gid];
    }
}