#if !defined(ENN_SOFTPLUS_ACTIVATION_H)
#define ENN_SOFTPLUS_ACTIVATION_H

#include <core/LayerBase.h>
#include <math.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class SoftplusActivation : public ActivationBase<T, T_SIZE> {
	const double ln1 = log(1);
public:
	inline virtual T forward(T val) const
	{
		return log(1.0 + exp((double)val));
	}

	inline virtual T backward(T val) const
	{
		double response = log(exp((double)val) - 1);
		return 1 / (1 + exp(-response));
	}
};

};

#endif
