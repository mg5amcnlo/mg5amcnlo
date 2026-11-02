#pragma once

#ifdef __CUDACC__

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuSetDevice cudaSetDevice
#define gpuMalloc cudaMalloc
#define gpuMallocAsync cudaMallocAsync
#define gpuFree cudaFree
#define gpuFreeAsync cudaFreeAsync
#define gpuMemcpy cudaMemcpy
#define gpuMemset cudaMemset
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemsetAsync cudaMemsetAsync
#define gpuStreamPerThread cudaStreamPerThread
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStream_t cudaStream_t
#define gpuEvent_t cudaEvent_t
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuStreamWaitEvent cudaStreamWaitEvent
#define gpuEventRecord cudaEventRecord
#define gpuEventQuery cudaEventQuery
#define gpuEventSynchronize cudaEventSynchronize
#define gpuErrorNotReady cudaErrorNotReady
#define gpuDeviceSynchronize cudaDeviceSynchronize

#define gpublasStatus_t cublasStatus_t
#define gpublasHandle_t cublasHandle_t
#define gpublasGetStatusString cublasGetStatusString
#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define gpublasCreate cublasCreate
#define gpublasDestroy cublasDestroy
#define gpublasSetStream cublasSetStream
#define gpublasDgemm cublasDgemm
#define gpublasDgemv cublasDgemv
#define gpublasSgemm cublasSgemm
#define gpublasSgemv cublasSgemv
#define GPUBLAS_OP_N CUBLAS_OP_N
#define GPUBLAS_OP_T CUBLAS_OP_T

#define thrust_par thrust::cuda::par

#elif defined __HIPCC__

#include <hip/hip_runtime_api.h>
#include <hipcub/hipcub.hpp>
#include <rocblas/rocblas.h>

#define gpuGetDeviceCount hipGetDeviceCount
#define gpuSetDevice hipSetDevice
#define gpuMalloc hipMalloc
#define gpuMallocAsync hipMallocAsync
#define gpuFree hipFree
#define gpuFreeAsync hipFreeAsync
#define gpuMemcpy hipMemcpy
#define gpuMemset hipMemset
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemsetAsync hipMemsetAsync
#define gpuStreamPerThread hipStreamPerThread
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuStreamWaitEvent(stream, event) hipStreamWaitEvent(stream, event, 0)
#define gpuEventRecord hipEventRecord
#define gpuEventQuery hipEventQuery
#define gpuEventSynchronize hipEventSynchronize
#define gpuErrorNotReady hipErrorNotReady
#define gpuDeviceSynchronize hipDeviceSynchronize

#define gpublasStatus_t rocblas_status
#define gpublasHandle_t rocblas_handle
#define gpublasGetStatusString rocblas_status_to_string
#define GPUBLAS_STATUS_SUCCESS rocblas_status_success
#define gpublasCreate rocblas_create_handle
#define gpublasDestroy rocblas_destroy_handle
#define gpublasSetStream rocblas_set_stream
#define gpublasDgemm rocblas_dgemm
#define gpublasDgemv rocblas_dgemv
#define gpublasSgemm rocblas_sgemm
#define gpublasSgemv rocblas_sgemv
#define GPUBLAS_OP_N rocblas_operation_none
#define GPUBLAS_OP_T rocblas_operation_transpose

#define thrust_par thrust::hip_rocprim::par
namespace cub = hipcub;

#endif

struct gpu_less_double {
    __host__ __device__ bool operator()(double a, double b) const { return a < b; }
};
