#if !defined(ENN_MVO_CONV_H)
#define ENN_MVO_CONV_H

#include <stdlib.h>
#include <math.h>
#include <limits>

namespace EasyNeuralNetworks {


template<typename T, typename T_SIZE, bool TRANSPOSED>
void convolve_1d_add(T * dst, const T * vec, const T * kernel, T_SIZE N, T_SIZE M, T_SIZE stride) {
	// vector is N
	// kernel is M
	// FIXME use fast algorithm for forward calculation
	T_SIZE j, i;
	if (!TRANSPOSED) {
		// DSTj = SUMi VEC[i + j * stride] * KERNEL[i]
		const T_SIZE dst_size = (N - M) / stride + 1;

		for (i = 0; i < dst_size; i++) {
			*dst += dot_product<T, T_SIZE>(vec, kernel, M);
			vec += stride;
			++dst;
		}
	} else {
		// DSTj = SUMi VEC[i - K + 1 + j * stride] * KERNEL[K-i-1]
		for (i = 0; i < N; i++) {
			T * d = dst + i * stride;
			const T * w = kernel;
			for (j = 0; j < M; j++) {
				*d += *w * *vec;
				++d;
				++w;
			}
			++vec;
		}
	}
}

template<typename T, typename T_SIZE, bool TRANSPOSED>
void convolve_2d_add(T * dst, const T * mat, const T * kernel, T_SIZE N, T_SIZE M, T_SIZE K, T_SIZE L, T_SIZE stride) {
	// matrix is NxM
	// kernel is KxL
	//
	// DSTab = SUMij MAT[i+j*N + a + b*N] * KERNEL[i + j*K] + KERNEL[K*L]{if BIAS};   a < N - K, b < M - L
	// FIXME use fast algorithm for forward calculation
	if (!TRANSPOSED) {
		T_SIZE a, b, j;
		const T * p;
		const T * k;
		T acc;
		const T_SIZE MLS = (M - L) / stride + 1;
		const T_SIZE NKS = (N - K) / stride + 1;

		for (b = 0; b < MLS; b++) {
			p = mat + (b * stride) * N;
			for (a = 0; a < NKS; a++) {
				*dst += dot_product_2d<T, T_SIZE>(p, kernel, N, M, K, L);
				p += N;
				++dst;
			}
		}
	} else {

	}
}

};

#endif
