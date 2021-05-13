#pragma once
#include "bitwizeNot_cuda.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/cudev.hpp>
#include <cuda_runtime.h>

//for __syncthreads() 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__ 
#endif //!(__CUDACC_RTC__) 

#include <device_launch_parameters.h>

__global__ void myKernel(cv::cudev::PtrStepSz<uchar3> dst) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < dst.cols) && (y < dst.rows)) {
		uchar b = UCHAR_MAX - dst.ptr(y)[x].x;
		uchar g = UCHAR_MAX - dst.ptr(y)[x].y;
		uchar r = UCHAR_MAX - dst.ptr(y)[x].z;

		dst.ptr(y)[x] = make_uchar3(b, g, r);
	}
}


void bitwiseNotCuda(cv::cuda::GpuMat &mat) {
	const dim3 block(32, 8);
	const dim3 grid(cv::cudev::divUp(mat.cols, block.x), cv::cudev::divUp(mat.rows, block.y));



	// 自作CUDAカーネルを呼び出す
	myKernel << <grid, block >> > (mat);

	CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}