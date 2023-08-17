#pragma once

//To provide a backward compatible __ldg() implementation. 
//Check this out: https://github.com/BryanCatanzaro/generics
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 320

template<typename T>
static __device__ __inline__ T __ldg(T* p){return *p;}

#endif

//This bypasses the L1 cache to use the L2 cache. This should be used for read only data only.
//The L2 cache fetches 32byte blocks instead of 128byte blocks.
template<typename T>
static __device__ __inline__ T ldg(T* p){
	return __ldg(p);
}


