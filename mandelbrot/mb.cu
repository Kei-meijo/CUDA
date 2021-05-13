#pragma once
#include "mb.cuh"
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

#include "mb_param.h"



__global__ void myKernel(cv::cudev::PtrStepSz<uchar3> dst) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < dst.cols) && (y < dst.rows)) {
		
		//原点座標
		double oy = (double)(dst.rows) / 2.0;
		double ox = (double)(dst.cols) / 3.0;

		//基準となる複素数
		double nx = ((double)x - 2 * ox) / ox;
		double ny = ((double)y - oy) / oy;

		//数列計算結果の複素数
		double zx = 0;
		double zy = 0;

		//収束したか
		double convergenceDecision = true;

		//計算用テンプレート
		double tx = 0;
		double ty = 0;

		//発散速度（大きいほど早く発散）
		int n = 0;
		
		for (int i = 0; i <= INF; i++) {

			double a = sqrt(zx * zx + zy * zy);
			if (a >= LIMIT) {
				convergenceDecision = false;
				break;
			}

			tx = zx * zx - zy * zy + nx;
			ty = 2 * zx * zy + ny;

			zx = tx;
			zy = ty;

			n = i;
		}

		//発散速度に応じて色を変える
		int r = (n % (INF / 11)) * 20;
		int g = (n % (INF / 15)) * 15;
		int b = (n % (INF / 19)) * 12;

		if (convergenceDecision) {
			//収束したら何もしない（色が黒）
		} else {
			dst.ptr(y)[x] = make_uchar3(b, g, r);
		}
	}
	//__syncthreads();
}


void createMB(cv::cuda::GpuMat &mat) {
	const dim3 block(32, 8);
	const dim3 grid(cv::cudev::divUp(mat.cols, block.x), cv::cudev::divUp(mat.rows, block.y));



	// 自作CUDAカーネルを呼び出す
	myKernel << <grid, block >> > (mat);

	CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}