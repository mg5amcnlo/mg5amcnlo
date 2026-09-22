/*
 * memory.h
 *
 *  Created on: 19.11.2020
 *      Author: shageboeck
 */

#ifndef MEMORY_H
#define MEMORY_H

#include "mgOnGpuTypes.h"

#include <memory>

template<typename T = fptype>
struct CudaHstDeleter {
  void operator()(T* mem) {
    checkCuda( cudaFreeHost( mem ) );
  }
};

#ifdef __CUDACC__
template<typename T = fptype>
struct CudaDevDeleter {
  void operator()(T* mem) {
    checkCuda( cudaFree( mem ) );
  }
};

template<typename T = fptype>
using unique_ptr_dev = std::unique_ptr<T, CudaDevDeleter<T>>;

template<typename T = fptype>
unique_ptr_dev<T> devMakeUnique(std::size_t N) {
  T* tmp = nullptr;
  checkCuda( cudaMalloc( &tmp, N * sizeof(T) ) );
  return std::unique_ptr<T, CudaDevDeleter<T>>{ tmp };
}

template<typename T = fptype>
using unique_ptr_host = std::unique_ptr<T[], CudaHstDeleter<T>>;

template<typename T = fptype>
unique_ptr_host<T> hstMakeUnique(std::size_t N) {
  T* tmp = nullptr;
  checkCuda( cudaMallocHost( &tmp, N * sizeof(T) ) );
  return std::unique_ptr<T[], CudaHstDeleter<T>>{ tmp };
};
#else
template<typename T = fptype>
using unique_ptr_host = std::unique_ptr<T[]>;
template<typename T = fptype>
std::unique_ptr<T[]> hstMakeUnique(std::size_t N) { return std::unique_ptr<T[]>{ new T[N] }; };
#endif



#endif /* MEMORY_H */
