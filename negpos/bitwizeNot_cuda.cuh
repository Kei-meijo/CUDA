#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

void bitwiseNotCuda(cv::cuda::GpuMat &mat);