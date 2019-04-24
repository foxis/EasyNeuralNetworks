#if !defined(ENN_TANH_ACTIVATION_H)
#define ENN_TANH_ACTIVATION_H

#include <LayerBase.h>
#include <math.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE>
class TanhActivation : public ActivationBase<T> {
public:
	inline virtual T forward(T val) const
	{
		return tanh(val);
	}

	inline virtual T backward(T val) const
	{
		T t = tanh(val);
		return 1 - t * t;
	}
};

};

#endif
