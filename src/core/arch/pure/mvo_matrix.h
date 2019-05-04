#if !defined(ENN_MVO_MATRIX_H)
#define ENN_MVO_MATRIX_H

#include <stdlib.h>
#include <math.h>
#include <limits>

namespace EasyNeuralNetworks {

template<typename T, typename T_SIZE>
inline T min_mat(T_SIZE * index_x, T_SIZE index_y, const T * a, T_SIZE in_width, T_SIZE width, T_SIZE height, T_SIZE stride = 1) {
	T acc = std::numeric_limits<T>::infinity();
	T_SIZE x = 0, y = 0;
	for (T_SIZE i = 0; i < height; i++) {
		T * p = *a;
		for (T_SIZE j = 0; j < width; j++) {
			auto tmp = *p;
			if (tmp < acc) {
				acc = tmp;
				x = j;
				y = i;
			}
			p += stride;
		}
		a += in_width;
	}
	*index_x = x;
	*index_y = y;
	return acc;
}

template<typename T, typename T_SIZE>
inline T max_mat(T_SIZE * index_x, T_SIZE index_y, const T * a, T_SIZE in_width, T_SIZE width, T_SIZE height, T_SIZE stride = 1) {
	T acc = -std::numeric_limits<T>::infinity();
	T_SIZE x = 0, y = 0;
	for (T_SIZE i = 0; i < height; i++) {
		T * p = *a;
		for (T_SIZE j = 0; j < width; j++) {
			auto tmp = *p;
			if (tmp > acc) {
				acc = tmp;
				x = j;
				y = i;
			}
			p += stride;
		}
		a += in_width;
	}
	*index_x = x;
	*index_y = y;
	return acc;
}

template<typename T, bool BIAS, typename T_SIZE, bool TRANSPOSED>
inline void mat_mul(T * dst, const T * vec, const T * mat, T_SIZE N, T_SIZE M) {
	// vector is N
	// matrix is NxM
	// destination is M
	// MATij = mat[i + j * (N + BIAS)]
	T acc;
	T_SIZE i, j;
	const T * v;

	if (!TRANSPOSED) {
		// DSTj = SUMi VECi * MATij + MAT(N+1)j {if BIAS=true}
		for (j = 0; j < M; j++) {
			acc = 0;
			for (i = 0, v = vec; i < N; i++) {
				acc += *v * *mat;
				++v;
				++mat;
			}
			if (BIAS) {
				acc += *mat;
				++mat;
			}
			*dst = acc;
			++dst;
		}
	} else {
		// DSTj = SUMi VECi * MATji
		const T * m;

		for (j = 0; j < N; j++) {
			acc = 0;
			m = mat + j;
			for (i = 0, v = vec; i < M; i++) {
				acc += *v * *m;
				++v;
				if (BIAS)
					m += N + 1;
				else
					m += N;
			}
			*dst = acc;
			++dst;
		}
	}
}

};

#endif
