#if !defined(ENN_MVO_VECTOR_H)
#define ENN_MVO_VECTOR_H

#include <stdlib.h>
#include <math.h>
#include <limits>

namespace EasyNeuralNetworks {

template<typename T, typename T_SIZE>
inline void hadamard_product(T * dst, const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	while (num--) {
		*dst = *a * *b;
		a += stridea;
		++dst;
		b += strideb;
	}
}

template<typename T, typename T_SIZE>
inline void hadamard_product_add(T * dst, const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	while (num--) {
		*dst += *a * *b;
		a += stridea;
		++dst;
		b += strideb;
	}
}

template<typename T, typename T_SIZE>
inline void normalize_vec(T * dst, T_SIZE num, T_SIZE stride) {
	T sum = sqrt(sqrsum_arr(dst, num, stride));
	while (num--) {
		*dst /= sum;
		dst += stride;
	}
}

// dot product of two vectors
template<typename T, typename T_SIZE>
inline T dot_product(const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	T acc = 0;
	while (num--) {
		acc += *a * *b;
		a += stridea;
		b += strideb;
	}
	return acc;
}

// dot product of KxL sumatrix of a and b of KxL
template<typename T, typename T_SIZE>
inline T dot_product_2d(const T * a, const T * b, T_SIZE N, T_SIZE M, T_SIZE K, T_SIZE L) {
	// a is NxM
	// b is KxL
	T acc = 0;
	T_SIZE num = K * L;
	for (T_SIZE j = 0; j < L; j++) {
		acc += dot_product(a, b, K);
		a += N;
		b += K;
	}

	return acc;
}

///
/// in two vectors, output - matrix
///
template<typename T, bool BIAS, typename T_SIZE>
inline void outer_product(T * dst, const T * u, const T * v, T_SIZE N, T_SIZE M) {
	const T * bp;

	for (T_SIZE i = 0; i < N; i++) {
		bp = v;
		for (T_SIZE j = 0; j < M; j++) {
				*dst = *u * *bp;
				++bp;
				++dst;
		}
		// update bias
		if (BIAS) {
			*dst = *u;
			++dst;
		}
		++u;
	}
}

template<typename T, bool BIAS, typename T_SIZE>
inline void outer_product_const(T * dst, const T * u, const T * v, T_SIZE N, T_SIZE M, T alpha) {
	const T * bp;

	for (T_SIZE i = 0; i < N; i++) {
		bp = v;
		for (T_SIZE j = 0; j < M; j++) {
				*dst = alpha * *u * *bp;
				++bp;
				++dst;
		}
		// update bias
		if (BIAS) {
			*dst = alpha * *u;
			++dst;
		}
		++u;
	}
}

template<typename T, bool BIAS, typename T_SIZE>
inline void outer_product_add_const(T * dst, const T * u, const T * v, T_SIZE N, T_SIZE M, T alpha) {
	const T * bp;

	for (T_SIZE i = 0; i < N; i++) {
		bp = v;
		for (T_SIZE j = 0; j < M; j++) {
				*dst += alpha * *u * *bp;
				++bp;
				++dst;
		}
		// update bias
		if (BIAS) {
			*dst += alpha * *u;
			++dst;
		}
		++u;
	}
}

};

#endif
