#if !defined(ENN_SOFTPLUS_ACTIVATION_H)
#define ENN_SOFTPLUS_ACTIVATION_H

#include <LayerBase.h>
#include <math.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE>
class SoftplusActivation : public ActivationBase<T> {
public:
	inline virtual T forward(T val) const
	{
		return log(1.0 + exp((double)val));
	}

	inline virtual T backward(T val) const
	{
		return 1.0 / (1.0 + exp(-(double)val));
	}
};

};

#endif