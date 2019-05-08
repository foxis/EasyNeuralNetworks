#if !defined(ENN_LOSS_H)
#define ENN_LOSS_H

#include <assert.h>
#include "tensor.h"
#include "matvecop.h"

namespace EasyNeuralNetworks {

#define ENN_T_LOSS_TYPEDEF(T_LOSS_NAME) typedef LossFunctionBase<T, T_SIZE> T_LOSS_NAME;

///
/// Abstract base class for loss functions
///
template<typename T, typename T_SIZE>
class LossFunctionBase {
public:
	virtual T operator () (tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& target, const tensor<T, T_SIZE>& output) const = 0;

	#define ENN_LOSS_LOOP(DELTA, ACC) \
		T acc = 0;	\
		assert(deltas.size() == target.size() && deltas.size() == output.size()); \
		auto num = output.size(); \
		auto D = deltas.data(); \
		auto Ta = target.data(); \
		auto O = output.data(); \
		while (num--) {	\
			*D = DELTA;	\
			acc += ACC;	\
			++Ta;	\
			++O;	\
			++D;	\
		}	\
		return acc;
};

///
/// MSE loss function
///
template<typename T, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class L2Loss : public LossFunctionBase<T, T_SIZE> {
public:
	virtual T operator () (tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& target, const tensor<T, T_SIZE>& output) const {
		ENN_LOSS_LOOP(*O - *Ta, *D * *D)
	}
};

///
/// Absolute error loss function
///
template<typename T, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class L1Loss : public LossFunctionBase<T, T_SIZE> {
public:
	virtual T operator () (tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& target, const tensor<T, T_SIZE>& output) const {
		ENN_LOSS_LOOP((*Ta < *O ? 1 : *Ta > *O ? -1 : 0), abs(*D))
	}
};

///
/// Cross Entropy loss function
///
template<typename T, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class CrossEntropy : public LossFunctionBase<T, T_SIZE> {
public:
	virtual T operator () (tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& target, const tensor<T, T_SIZE>& output) const {
		ENN_LOSS_LOOP(-*Ta * log10(*O), -*D)
	}
};

};

#endif
