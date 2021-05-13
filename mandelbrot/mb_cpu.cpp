#include <omp.h>
#include "mb_cpu.h"
#include "mb_param.h"

void createMBCPU(cv::Mat& mat) {

	//���_���W
	double oy = (double)(mat.rows) / 2.0;
	double ox = (double)(mat.cols) / 3.0;

//#pragma omp parallel for
	for (int y = 0; y < mat.rows; y++) {
		cv::Vec3b *p = mat.ptr<cv::Vec3b>(y);
		for (int x = 0; x < mat.cols; x++) {
			//��ƂȂ镡�f��
			double nx = ((double)x - 2 * ox) / ox;
			double ny = ((double)y - oy) / oy;

			//����v�Z���ʂ̕��f��
			double zx = 0;
			double zy = 0;

			//����������
			double convergenceDecision = true;

			//�v�Z�p�e���v���[�g
			double tx = 0;
			double ty = 0;

			//���U���x�i�傫���قǑ������U�j
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

			//���U���x�ɉ����ĐF��ς���
			int r = (n % (INF / 11)) * 20;
			int g = (n % (INF / 15)) * 15;
			int b = (n % (INF / 19)) * 12;

			if (convergenceDecision) {
				//���������牽�����Ȃ��i�F�����j
			} else {
				p[x][0] = b;
				p[x][1] = g;
				p[x][2] = r;
			}
		}
	}
	
}