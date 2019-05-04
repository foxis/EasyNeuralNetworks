#if !defined(ENN_MVO_ARRAY_H)
#define ENN_MVO_ARRAY_H

#include <stdlib.h>
#include <math.h>
#include <limits>

namespace EasyNeuralNetworks {

template<typename T, typename T_SIZE>
inline void diff_arr(T * dst, const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	while (num--) {
		*dst = *a - *b;
		a += stridea;
		++dst;
		b += strideb;
	}
}

template<typename T, typename T_SIZE>
inline void sum_arr(T * dst, const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	while (num--) {
		*dst = *a + *b;
		a += stridea;
		++dst;
		b += strideb;
	}
}

template<typename T, typename T_SIZE>
inline void sum_arr(T * dst, const T * src, T c, T_SIZE num, T_SIZE stride = 1) {
	while (num--) {
		*dst = *src + c;
		src += stride;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void diff_arr(T * dst, const T * src, T c, T_SIZE num, T_SIZE stride = 1) {
	while (num--) {
		*dst = *src - c;
		src += stride;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void mul_arr(T * dst, const T * src, T c, T_SIZE num, T_SIZE stride = 1) {
	while (num--) {
		*dst = *src * c;
		src += stride;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void div_arr(T * dst, const T * src, T c, T_SIZE num, T_SIZE stride = 1) {
	while (num--) {
		*dst = *src / c;
		src += stride;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void sum_arr_add(T * dst, const T * src, T c, T_SIZE num, T_SIZE stride = 1) {
	while (num--) {
		*dst += *src + c;
		src += stride;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void diff_arr_add(T * dst, const T * src, T c, T_SIZE num, T_SIZE stride = 1) {
	while (num--) {
		*dst += *src - c;
		src += stride;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void mul_arr_add(T * dst, const T * src, T c, T_SIZE num, T_SIZE stride = 1) {
	while (num--) {
		*dst += *src * c;
		src += stride;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void div_arr_add(T * dst, const T * src, T c, T_SIZE num, T_SIZE stride = 1) {
	while (num--) {
		*dst += *src / c;
		src += stride;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void sum_arr(T * dst, T c, T_SIZE num) {
	while (num--) {
		*dst += c;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void diff_arr(T * dst, T c, T_SIZE num) {
	while (num--) {
		*dst -= c;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void mul_arr(T * dst, T c, T_SIZE num) {
	while (num--) {
		*dst *= c;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void div_arr(T * dst, T c, T_SIZE num) {
	while (num--) {
		*dst /= c;
		++dst;
	}
}

template<typename T, typename T_SIZE>
inline void diffsqr_arr(T * dst, const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	T tmp;
	while (num--) {
		tmp = *a - *b
		*dst = tmp * tmp;
		a += stridea;
		++dst;
		b += strideb;
	}
}

template<typename T, typename T_SIZE>
inline void sumsqr_arr(T * dst, const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	T tmp;
	while (num--) {
		tmp = *a + *b
		*dst = tmp * tmp;
		a += stridea;
		++dst;
		b += strideb;
	}
}

template<typename T, typename T_SIZE>
inline void sqrdiff_arr(T * dst, const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	while (num--) {
		*dst = *a **a - *b **b;
		a += stridea;
		++dst;
		b += strideb;
	}
}

template<typename T, typename T_SIZE>
inline void sqrsum_arr(T * dst, const T * a, const T * b, T_SIZE num, T_SIZE stridea = 1, T_SIZE strideb = 1) {
	while (num--) {
		*dst = *a * *a + *b * *b;
		a += stridea;
		++dst;
		b += strideb;
	}
}

template<typename T, typename T_SIZE>
inline T sum_arr(const T * a, T_SIZE num, T_SIZE stride = 1) {
	T acc = 0;
	while (num--) {
		acc += *a;
		a += stride;
	}
	return acc;
}

template<typename T, typename T_SIZE>
inline T sqrsum_arr(const T * a, T_SIZE num, T_SIZE stride = 1) {
	T acc = 0;
	while (num--) {
		acc += *a * *a;
		a += stride;
	}
	return acc;
}

template<typename T, typename T_SIZE>
inline T min_arr(T_SIZE * index, const T * a, T_SIZE num, T_SIZE stride = 1) {
	T acc = std::numeric_limits<T>::infinity();
	T_SIZE idx = 0;
	for (T_SIZE i = 0; i < num; i++) {
		auto tmp = *a;
		if (tmp < acc) {
			acc = tmp;
			idx = i;
		}
		a += stride;
	}
	if (index != NULL)
		*index = idx;
	return acc;
}

template<typename T, typename T_SIZE>
inline T max_arr(T_SIZE * index, const T * a, T_SIZE num, T_SIZE stride = 1) {
	T acc = -std::numeric_limits<T>::infinity();
	T_SIZE idx = 0;
	for (T_SIZE i = 0; i < num; i++) {
		auto tmp = *a;
		if (tmp > acc) {
			acc = tmp;
			idx = i;
		}
		a += stride;
	}
	if (index != NULL)
		*index = idx;
	return acc;
}

template<typename T, typename T_SIZE>
inline T mean_arr(const T * a, T_SIZE num, T_SIZE stride = 1) {
	return sum_arr(a, num, stride) / (T)num;
}

template<typename T, typename T_SIZE>
inline void moments_arr(T * mean, T * stddev, const T * a, T_SIZE num, T_SIZE stride = 1) {
	if (num < 2)
		return ;

	T K = a[0];
	T n = num;
	T Ex = 0;
	T Ex2 = 0.0;
	T tmp;

	while (num--) {
		tmp = *a - K;
		Ex += tmp;
		Ex2 += tmp * tmp;
		a += stride;
	}
	*mean = K + Ex / n;
	*stddev = (Ex2 - (Ex * Ex) / n) / n;
}

///
/// weighted moments
///
template<typename T, typename T_SIZE>
inline void moments_arr(T * mean, T * stddev, const T * a, const T * w, T_SIZE num, T_SIZE stride = 1) {
	if (num < 2)
		return ;

	T K = *a * *w;
	T n = num;
	T Ex = 0;
	T Ex2 = 0.0;
	T tmp;

	while (num--) {
		tmp = *a * *w - K;
		Ex += tmp;
		Ex2 += tmp * tmp;
		a += stride;
		w += stride;
	}
	*mean = K + Ex / n;
	*stddev = (Ex2 - (Ex * Ex) / n) / n;
}

};

#endif
