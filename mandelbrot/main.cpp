#include <opencv2/opencv.hpp>
#include <windows.h>

#include "mb.cuh"
#include "mb_cpu.h"

int main() {

	cv::Mat output_cpu, output_gpu;

	int size = 4;//4K
	int width = 1920 * size / 2;
	int height = 1080 * size / 2;
	cv::Size winSize(width, height);

	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	LARGE_INTEGER start, end;


	//CPU
	QueryPerformanceCounter(&start);
	output_cpu = cv::Mat(winSize, CV_8UC3);
	createMBCPU(output_cpu);
	QueryPerformanceCounter(&end);

	double time_cpu = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	std::cout << "CPU : " << time_cpu << "ms" << std::endl;

	//GPU
	QueryPerformanceCounter(&start);
	
	cv::cuda::GpuMat gpu(winSize, CV_8UC3);
	createMB(gpu);

	gpu.download(output_gpu);

	QueryPerformanceCounter(&end);

	double time_gpu = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	std::cout << "GPU : " << time_gpu << "ms" << std::endl;

	cv::imshow("CPU", output_cpu);
	cv::imshow("GPU", output_gpu);

	cv::waitKey(0);
	return EXIT_SUCCESS;
}