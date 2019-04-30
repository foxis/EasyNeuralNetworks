#if !defined(ENN_TANH_ACTIVATION_H)
#define ENN_TANH_ACTIVATION_H

#include <core/LayerBase.h>
#include <math.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class TanhActivation : public ActivationBase<T, T_SIZE> {
public:
	inline virtual T forward(T val) const
	{
		return tanh((double)val);
	}

	inline virtual T backward(T val) const
	{
		return (T)1 - val * val;
	}
};

};

#endif
