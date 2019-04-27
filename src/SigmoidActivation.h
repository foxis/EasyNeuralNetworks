#if !defined(ENN_SIGMOID_ACTIVATION_H)
#define ENN_SIGMOID_ACTIVATION_H

#include <LayerBase.h>
#include <math.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class SigmoidActivation : public ActivationBase<T, T_SIZE> {
public:
	inline virtual T forward(T val) const
	{
		return 1.0 / (1.0 + exp(-(double)val));
	}

	inline virtual T backward(T val) const
	{
		return val * ((T)1 - val);
	}
};

};

#endif
