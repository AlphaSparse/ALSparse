#ifndef AMGX_UTIL_H
#define AMGX_UTIL_H

#include "cuda.h"

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif

// namespace utils
// {

#define DEFAULT_MASK 0xffffffff
#define warpSize 32
// ====================================================================================================================
// Shuffle.
// ====================================================================================================================
static __device__ __forceinline__ int shfl( int r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    return __shfl_sync( mask, r, lane, bound );}


static __device__ __forceinline__ float shfl( float r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    return __shfl_sync( mask, r, lane, bound );
}

static __device__ __forceinline__ double shfl( double r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    return __shfl_sync(mask, r, lane, bound );
}

static __device__ __forceinline__ cuComplex shfl( cuComplex r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    float re = __shfl_sync( mask, cuCrealf(r), lane, bound );
    float im = __shfl_sync( mask, cuCimagf(r), lane, bound );
    return make_cuComplex(re, im);
}

static __device__ __forceinline__ cuDoubleComplex shfl( cuDoubleComplex r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    double re = shfl( cuCreal(r), lane, mask, bound );
    double im = shfl( cuCimag(r), lane, mask, bound );
    return make_cuDoubleComplex( re, im );
}

static __device__ __forceinline__ int shfl_xor( int r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    return __shfl_xor_sync( mask, r, lane_mask, bound );
}


static __device__ __forceinline__ float shfl_xor( float r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    return __shfl_xor_sync( mask, r, lane_mask, bound );
}

static __device__ __forceinline__ double shfl_xor( double r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    return __shfl_xor_sync( mask, r, lane_mask, bound );
}

static __device__ __forceinline__ cuComplex shfl_xor( cuComplex r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    float re = __shfl_xor_sync( mask, cuCrealf(r), lane_mask, bound );
    float im = __shfl_xor_sync( mask, cuCimagf(r), lane_mask, bound );
    return make_cuComplex(re, im);
}

static __device__ __forceinline__ cuDoubleComplex shfl_xor( cuDoubleComplex r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    double re = shfl_xor( cuCreal(r), lane_mask, mask, bound );
    double im = shfl_xor( cuCimag(r), lane_mask, mask, bound );
    return make_cuDoubleComplex( re, im );
}

static __device__ __forceinline__ unsigned int u_any(int p, unsigned int mask = DEFAULT_MASK)
{
    return __any_sync(mask, p);
}

static __device__ __forceinline__ unsigned int all(int p, unsigned int mask = DEFAULT_MASK)
{
    return __all_sync(mask, p);
}

static __device__ __forceinline__ unsigned int u_ballot(int p, unsigned int mask = DEFAULT_MASK)
{
    return __ballot_sync(mask, p);
}

// ====================================================================================================================
// Loads.
// ====================================================================================================================

enum Ld_mode { LD_AUTO = 0, LD_CA, LD_CG, LD_TEX, LD_NC };

template< Ld_mode Mode >
struct Ld {};

template<>
struct Ld<LD_AUTO>
{
    template< typename T >
    static __device__ __forceinline__ T load( const T *ptr ) { return *ptr; }
};

template<>
struct Ld<LD_CG>
{
    static __device__ __forceinline__ int load( const int *ptr )
    {
        int ret;
        asm volatile ( "ld.global.cg.s32 %0, [%1];"  : "=r"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ float load( const float *ptr )
    {
        float ret;
        asm volatile ( "ld.global.cg.f32 %0, [%1];"  : "=f"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ double load( const double *ptr )
    {
        double ret;
        asm volatile ( "ld.global.cg.f64 %0, [%1];"  : "=d"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ cuComplex load( const cuComplex *ptr )
    {
        float ret[2];
        asm volatile ( "ld.global.cg.v2.f32 {%0, %1}, [%2];"  : "=f"(ret[0]), "=f"(ret[1]) : __PTR( (float *)(ptr) ) );
        return make_cuComplex(ret[0], ret[1]);
    }

    static __device__ __forceinline__ cuDoubleComplex load( const cuDoubleComplex *ptr )
    {
        double ret[2];
        asm volatile ( "ld.global.cg.v2.f64 {%0, %1}, [%2];"  : "=d"(ret[0]), "=d"(ret[1]) : __PTR( (double *)(ptr) ) );
        return make_cuDoubleComplex(ret[0], ret[1]);
    }

};

template<>
struct Ld<LD_CA>
{
    static __device__ __forceinline__ int load( const int *ptr )
    {
        int ret;
        asm volatile ( "ld.global.ca.s32 %0, [%1];"  : "=r"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ float load( const float *ptr )
    {
        float ret;
        asm volatile ( "ld.global.ca.f32 %0, [%1];"  : "=f"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ double load( const double *ptr )
    {
        double ret;
        asm volatile ( "ld.global.ca.f64 %0, [%1];"  : "=d"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ cuComplex load( const cuComplex *ptr )
    {
        float ret[2];
        asm volatile ( "ld.global.ca.v2.f32 {%0, %1}, [%2];"  : "=f"(ret[0]), "=f"(ret[1]) : __PTR( (float *)(ptr) ) );
        return make_cuComplex(ret[0], ret[1]);
    }

    static __device__ __forceinline__ cuDoubleComplex load( const cuDoubleComplex *ptr )
    {
        double ret[2];
        asm volatile ( "ld.global.ca.v2.f64 {%0, %1}, [%2];"  : "=d"(ret[0]), "=d"(ret[1]) : __PTR( (double *)(ptr) ) );
        return make_cuDoubleComplex(ret[0], ret[1]);
    }
};

template<>
struct Ld<LD_NC>
{
    template< typename T >
    static __device__ __forceinline__ T load( const T *ptr ) { return __ldg( ptr ); }
};

static __device__ __forceinline__ unsigned int activemask()
{
    return __activemask();
}

static __device__ __forceinline__ void syncwarp(unsigned int mask = 0xffffffff)
{
    return __syncwarp(mask);
}

static __device__ __forceinline__ void atomic_add( float *address, float value )
{
    atomicAdd( address, value );
}

static __device__ __forceinline__ void atomic_add( double *address, double value )
{
    atomicAdd( address, value );
}

static __device__ __forceinline__ void atomic_add( cuComplex *address, cuComplex value )
{
    atomicAdd((float *)(address), cuCrealf(value));
    atomicAdd((float *)((char *)(address) + sizeof(float)), cuCimagf(value));
}

static __device__ __forceinline__ void atomic_add( cuDoubleComplex *address, cuDoubleComplex value )
{
    atomic_add((double *)(address), cuCreal(value));
    atomic_add((double *)((char *)(address) + sizeof(double)), cuCimag(value));
}

static __device__ __forceinline__ int64_t atomic_CAS(int64_t* address, int64_t compare, int64_t val)
{
    return (int64_t)atomicCAS((unsigned long long *)address, (unsigned long long)compare, (unsigned long long)val);
}

static __device__ __forceinline__ int atomic_CAS(int* address, int compare, int val)
{
    return atomicCAS(address, compare, val);
}


static __device__ __forceinline__ int lane_id()
{
    int id;
    asm( "mov.u32 %0, %%laneid;" : "=r"(id) );
    return id;
}

static __device__ __forceinline__ int lane_mask_lt()
{
    int mask;
    asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(mask) );
    return mask;
}

static __device__ __forceinline__ int warp_id()
{
    return threadIdx.x >> 5;
}

__device__ __forceinline__ int get_work( int *queue, int warp_id )
{
    int offset = -1;

    if ( lane_id() == 0 )
    {
        offset = atomicAdd( queue, 1 );
    }

    return shfl( offset, 0 );
}

// }
#endif