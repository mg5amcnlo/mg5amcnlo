// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: J. Teig (Jul 2023) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2020-2025).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MG5AMC_GPUABSTRACTION_H
#define MG5AMC_GPUABSTRACTION_H 1

#include "mgOnGpuConfig.h"

#include <cassert>

//--------------------------------------------------------------------------

#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)

#ifndef MGONGPU_HAS_NO_BLAS
#include "cublas_v2.h"
#endif

#define gpuError_t cudaError_t
#define gpuPeekAtLastError cudaPeekAtLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess

#define gpuMallocHost( ptr, size ) checkGpu( cudaMallocHost( ptr, size ) )
#define gpuMalloc( ptr, size ) checkGpu( cudaMalloc( ptr, size ) )

#define gpuMemcpy( dstData, srcData, srcBytes, func ) checkGpu( cudaMemcpy( dstData, srcData, srcBytes, func ) )
#define gpuMemset( data, value, bytes ) checkGpu( cudaMemset( data, value, bytes ) )
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyToSymbol( type1, type2, size ) checkGpu( cudaMemcpyToSymbol( type1, type2, size ) )

#define gpuFree( ptr ) checkGpu( cudaFree( ptr ) )
#define gpuFreeHost( ptr ) checkGpu( cudaFreeHost( ptr ) )

#define gpuGetSymbolAddress( devPtr, symbol ) checkGpu( cudaGetSymbolAddress( devPtr, symbol ) )

#define gpuSetDevice cudaSetDevice
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuDeviceReset cudaDeviceReset

#define gpuLaunchKernel( kernel, blocks, threads, ... ) kernel<<<blocks, threads>>>( __VA_ARGS__ )
//#define gpuLaunchKernelSharedMem( kernel, blocks, threads, sharedMem, ... ) kernel<<<blocks, threads, sharedMem>>>( __VA_>
#define gpuLaunchKernelStream( kernel, blocks, threads, stream, ... ) kernel<<<blocks, threads, 0, stream>>>( __VA_ARGS__ )
#define gpuLaunchKernel2D( kernel, blocks_x, blocks_y, threads, stream, ... ) kernel<<<dim3(blocks_x, blocks_y, 1), dim3(threads, 1, 1), 0, stream>>>( __VA_ARGS__ )

#define gpuStream_t cudaStream_t
#define gpuStreamCreate( pStream ) checkGpu( cudaStreamCreate( pStream ) )
#define gpuStreamDestroy( stream ) checkGpu( cudaStreamDestroy( stream ) )
#define gpuMallocAsync( ptr, size, stream ) checkGpu( cudaMallocAsync( ptr, size, stream ) )
#define gpuFreeAsync( ptr, stream ) checkGpu( cudaFreeAsync( ptr, stream ) )
#define gpuStreamSynchronize( stream ) checkGpu( cudaStreamSynchronize( stream ) )

#define gpuBlasStatus_t cublasStatus_t
#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#ifndef MGONGPU_HAS_NO_BLAS
#define gpuBlasHandle_t cublasHandle_t
#else
#define gpuBlasHandle_t void // hack to keep the same API also in noBLAS builds
#endif
#define gpuBlasCreate cublasCreate
#define gpuBlasDestroy cublasDestroy
#define gpuBlasSetStream cublasSetStream

#define gpuBlasSaxpy cublasSaxpy
#define gpuBlasSdot cublasSdot
#define gpuBlasSgemv cublasSgemv
#define gpuBlasSgemm cublasSgemm
#define gpuBlasSgemmStridedBatched cublasSgemmStridedBatched
#define gpuBlasDaxpy cublasDaxpy
#define gpuBlasDdot cublasDdot
#define gpuBlasDgemv cublasDgemv
#define gpuBlasDgemm cublasDgemm
#define gpuBlasDgemmStridedBatched cublasDgemmStridedBatched
#define GPUBLAS_OP_N CUBLAS_OP_N
#define GPUBLAS_OP_T CUBLAS_OP_T

//--------------------------------------------------------------------------

#elif defined __HIPCC__

#ifndef MGONGPU_HAS_NO_BLAS
#include "hipblas/hipblas.h"
#endif

#define gpuError_t hipError_t
#define gpuPeekAtLastError hipPeekAtLastError
#define gpuGetErrorString hipGetErrorString
#define gpuSuccess hipSuccess

#define gpuMallocHost( ptr, size ) checkGpu( hipHostMalloc( ptr, size ) ) // HostMalloc better
#define gpuMalloc( ptr, size ) checkGpu( hipMalloc( ptr, size ) )

#define gpuMemcpy( dstData, srcData, srcBytes, func ) checkGpu( hipMemcpy( dstData, srcData, srcBytes, func ) )
#define gpuMemset( data, value, bytes ) checkGpu( hipMemset( data, value, bytes ) )
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyToSymbol( type1, type2, size ) checkGpu( hipMemcpyToSymbol( type1, type2, size ) )

#define gpuFree( ptr ) checkGpu( hipFree( ptr ) )
#define gpuFreeHost( ptr ) checkGpu( hipHostFree( ptr ) )

#define gpuGetSymbolAddress( devPtr, symbol ) checkGpu( hipGetSymbolAddress( devPtr, symbol ) )

#define gpuSetDevice hipSetDevice
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuDeviceReset hipDeviceReset

#define gpuLaunchKernel( kernel, blocks, threads, ... ) kernel<<<blocks, threads>>>( __VA_ARGS__ )
//#define gpuLaunchKernelSharedMem( kernel, blocks, threads, sharedMem, ... ) kernel<<<blocks, threads, sharedMem>>>( __VA_>
#define gpuLaunchKernelStream( kernel, blocks, threads, stream, ... ) kernel<<<blocks, threads, 0, stream>>>( __VA_ARGS__ )
#define gpuLaunchKernel2D( kernel, blocks_x, blocks_y, threads, stream, ... ) kernel<<<dim3(blocks_x, blocks_y, 1), dim3(threads, 1, 1), 0, stream>>>( __VA_ARGS__ )

#define gpuStream_t hipStream_t
#define gpuStreamCreate( pStream ) checkGpu( hipStreamCreate( pStream ) )
#define gpuStreamDestroy( stream ) checkGpu( hipStreamDestroy( stream ) )
#define gpuMallocAsync( ptr, size, stream ) checkGpu( hipMallocAsync( ptr, size, stream ) )
#define gpuFreeAsync( ptr, stream ) checkGpu( hipFreeAsync( ptr, stream ) )
#define gpuStreamSynchronize( stream ) checkGpu( hipStreamSynchronize( stream ) )

#define gpuBlasStatus_t hipblasStatus_t
#define GPUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#ifndef MGONGPU_HAS_NO_BLAS
#define gpuBlasHandle_t hipblasHandle_t
#else
#define gpuBlasHandle_t void // hack to keep the same API also in noBLAS builds
#endif
#define gpuBlasCreate hipblasCreate
#define gpuBlasDestroy hipblasDestroy
#define gpuBlasSetStream hipblasSetStream

#define gpuBlasSaxpy hipblasSaxpy
#define gpuBlasSdot hipblasSdot
#define gpuBlasSgemv hipblasSgemv
#define gpuBlasSgemm hipblasSgemm
#define gpuBlasSgemmStridedBatched hipblasSgemmStridedBatched
#define gpuBlasDaxpy hipblasDaxpy
#define gpuBlasDdot hipblasDdot
#define gpuBlasDgemv hipblasDgemv
#define gpuBlasDgemm hipblasDgemm
#define gpuBlasDgemmStridedBatched hipblasDgemmStridedBatched
#define GPUBLAS_OP_N HIPBLAS_OP_N
#define GPUBLAS_OP_T HIPBLAS_OP_T

#endif

//--------------------------------------------------------------------------

#ifdef MGONGPU_FPTYPE2_FLOAT
#define gpuBlasTaxpy gpuBlasSaxpy
#define gpuBlasTdot gpuBlasSdot
#define gpuBlasTgemv gpuBlasSgemv
#define gpuBlasTgemm gpuBlasSgemm
#define gpuBlasTgemmStridedBatched gpuBlasSgemmStridedBatched
#else
#define gpuBlasTaxpy gpuBlasDaxpy
#define gpuBlasTdot gpuBlasDdot
#define gpuBlasTgemv gpuBlasDgemv
#define gpuBlasTgemm gpuBlasDgemm
#define gpuBlasTgemmStridedBatched gpuBlasDgemmStridedBatched
#endif

#endif // MG5AMC_GPUABSTRACTION_H
