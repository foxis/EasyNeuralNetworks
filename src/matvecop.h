#if !defined(ENN_MATVECOP_H)
#define ENN_MATVECOP_H

#include <stdlib.h>
#include <math.h>

#define ENN_BIAS (BIAS?1:0)

namespace EasyNeuralNetworks {

/// ==========================================
/// per element operations
///
///
template<typename T, typename T_SIZE>
inline void diff_arr(T * dst, const T * a, const T * b, T_SIZE num) {
	for (T_SIZE i = 0; i < num; i++) {
		*dst = *a - *b;
		++a;
		++dst;
		++b;
	}
}

template<typename T, typename T_SIZE>
inline void sqrdiff_arr(T * dst, const T * a, const T * b, T_SIZE num) {
	T tmp;
	for (T_SIZE i = 0; i < num; i++) {
		tmp = *a - *b
		*dst = tmp * tmp;
		++a;
		++dst;
		++b;
	}
}

/// ==========================================
/// Accumulation methods
///
///
template<typename T, typename T_SIZE>
inline T dot_arr(const T * a, const T * b, T_SIZE num) {
	T acc = 0;
	for (T_SIZE i = 0; i < num; i++) {
		acc += *a * *b;
		++a;
		++b;
	}
	return acc;
}

template<typename T, typename T_SIZE>
inline T dot_arr(const T * a, const T * b, T_SIZE num, T_SIZE stridea, T_SIZE strideb) {
	T acc = 0;
	for (T_SIZE i = 0; i < num; i++) {
		acc += *a * *b;
		a += stridea;
		b += strideb;
	}
	return acc;
}

template<typename T, typename T_SIZE>
inline T sum_arr(const T * a, T_SIZE num) {
	T acc = 0;
	for (T_SIZE i = 0; i < num; i++) {
		acc += *a;
		++a;
	}
	return acc;
}

template<typename T, typename T_SIZE>
inline T sqrsum_arr(const T * a, T_SIZE num) {
	T acc = 0;
	for (T_SIZE i = 0; i < num; i++) {
		acc += *a * *a;
		++a;
	}
	return acc;
}

template<typename T, typename T_SIZE>
inline T sum_arr(const T * a, T_SIZE num, T_SIZE stride) {
	T acc = 0;
	for (T_SIZE i = 0; i < num; i++) {
		acc += *a;
		a += stride;
	}
	return acc;
}

template<typename T, typename T_SIZE>
inline T sqrsum_arr(const T * a, T_SIZE num, T_SIZE stride) {
	T acc = 0;
	for (T_SIZE i = 0; i < num; i++) {
		acc += *a * *a;
		a += stride;
	}
	return acc;
}

/// ==========================================
/// matrix operations
///
///
template<typename T, bool BIAS, typename T_SIZE, bool TRANSPOSED>
inline void mat_mul(T * dst, const T * vec, const T * mat, T_SIZE N, T_SIZE M) {
	// vector is N
	// matrix is (N+BIAS)xM
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

template<typename T, bool BIAS, typename T_SIZE, bool TRANSPOSED>
inline void convolve_1d(T * dst, const T * vec, const T * kernel, T_SIZE N, T_SIZE M, T_SIZE stride) {
	// vector is N
	// kernel is M
	if (!TRANSPOSED) {
		// DSTj = SUMi VEC[i+j] * KERNEL[i] + KERNEL[M]{if BIAS};
		T_SIZE j;
		const T_SIZE dst_size = (N - M) / stride + 1;

		for (j = 0; j < dst_size; j++) {
			*dst = dot_arr<T, T_SIZE>(vec, kernel, M);
			vec += stride;
			if (BIAS)
				*dst += kernel[M];
			++dst;
		}
	} else {
		// DSTj = SUMi VEC[i + j * ] * KERNEL[i]
		for (T_SIZE i = 0; i < N; i++) {
			T * d = dst + i * stride;
			const T * w = kernel;
			for (T_SIZE j = 0; j < M; j++) {
				*d += *w * *vec;
				++d;
				++w;
			}
			++vec;
		}
	}
}

template<typename T, bool BIAS, typename T_SIZE, bool TRANSPOSED>
inline void convolve_2d(T * dst, const T * mat, const T * kernel, T_SIZE N, T_SIZE M, T_SIZE K, T_SIZE L, T_SIZE stride) {
	// matrix is NxM
	// kernel is KxL
	//
	// DSTab = SUMij MAT[i+j*N + a + b*N] * KERNEL[i + j*K] + KERNEL[K*L]{if BIAS};   a < N - K, b < M - L
	if (!TRANSPOSED) {
		T_SIZE a, b;
		const T * p;
		const T_SIZE MLS = (M - L) / stride + 1;
		const T_SIZE NKS = (N - K) / stride + 1;
		const T_SIZE KL = K * L;

		for (b = 0; b < MLS; b++) {
			p = mat + (b * stride) * N;
			for (a = 0; a < NKS; a++) {
				*dst = dot_arr<T, BIAS, T_SIZE>(p, kernel, K * L, N, 1);
				p += stride;
				if (BIAS)
					*dst += kernel[KL];
				++dst;
			}
		}
	} else {
		
	}
}

/// ==========================================
/// misc methods
///
///
inline float random_flat(float mean, float scale) {
	return scale * ((float).5 - (rand() / (float)RAND_MAX) + mean);
}

inline float random_normal(float mean, float stddev) {
	//Box muller method(stackoverflow https://stackoverflow.com/questions/19944111/creating-a-gaussian-random-generator-with-a-mean-and-standard-deviation)
	static float n2 = 0.0;
	static int n2_cached = 0;
	if (!n2_cached)
	{
			float x, y, r;
			do
			{
					x = 2.0 * rand() / (float)RAND_MAX - 1;
					y = 2.0 * rand() / (float)RAND_MAX - 1;

					r = x * x + y * y;
			}
			while (r == 0.0 || r > 1.0);
			{
					float d = sqrt(-2.0 * log(r) / r);
					float n1 = x * d;
					n2 = y * d;
					float result = n1 * stddev + mean;
					n2_cached = 1;
					return result;
			}
	}
	else
	{
			n2_cached = 0;
			return n2 * stddev + mean;
	}
}

};

#endif
