#include <opencv2/opencv.hpp>
#include <windows.h>
#include <omp.h>

#include "bitwizeNot_cuda.cuh"

int main() {

	cv::Mat input, output_cpu, output_gpu;
	input = cv::imread("ImageName");

	if (input.empty()) {
		return EXIT_FAILURE;
	}

	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	LARGE_INTEGER start, end;

	//CPU
	QueryPerformanceCounter(&start);
	//cv::bitwise_not(input, output_cpu);
	output_cpu = cv::Mat(input.size(), CV_8UC3);
#pragma omp parallel for
	for (int y = 0; y < input.rows; y++) {
		cv::Vec3b *in = input.ptr<cv::Vec3b>(y);
		cv::Vec3b *out = output_cpu.ptr<cv::Vec3b>(y);
		for (int x = 0; x < input.cols; x++) {
			for (int c = 0; c < 3; c++) {
				out[x][c] = UCHAR_MAX - in[x][c];
			}
		}
	}
	QueryPerformanceCounter(&end);

	double time_cpu = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	std::cout << "CPU : " << time_cpu << "ms" << std::endl;

	//GPU
	QueryPerformanceCounter(&start);
	
	cv::cuda::GpuMat gpu(input);
	bitwiseNotCuda(gpu);
	gpu.download(output_gpu);

	QueryPerformanceCounter(&end);

	double time_gpu = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	std::cout << "GPU : " << time_gpu << "ms" << std::endl;

	cv::imshow("CPU", output_cpu);
	cv::imshow("GPU", output_gpu);

	cv::waitKey(0);
	return EXIT_SUCCESS;
}