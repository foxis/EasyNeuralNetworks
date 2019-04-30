#if !defined(ENN_SOFTPLUS_ACTIVATION_H)
#define ENN_SOFTPLUS_ACTIVATION_H

#include <core/LayerBase.h>
#include <math.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class SoftplusActivation : public ActivationBase<T, T_SIZE> {
	const double ln1 = ln(1);
public:
	inline virtual T forward(T val) const
	{
		return ln(1.0 + exp((double)val));
	}

	inline virtual T backward(T val) const
	{
		return 1.0 / (1.0 + exp(-(double)val / ln1));
	}
};

};

#endif
