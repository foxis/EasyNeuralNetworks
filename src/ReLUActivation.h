#if !defined(ENN_RELU_ACTIVATION_H)
#define ENN_RELU_ACTIVATION_H

#include <LayerBase.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE>
class ReLUActivation : public ActivationBase<T> {
	T _negative_d;
public:
	ReLUActivation() { _negative_d = 0; }
	ReLUActivation(T negative_d) { _negative_d = negative_d; }

	inline virtual T forward(T val) const
	{
		return val >= 0 ? val : _negative_d * val;
	}

	inline virtual T backward(T val) const
	{
		return val < 0 ? _negative_d : 1;
	}
};

};

#endif
