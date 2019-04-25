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
		return tanh((double)val);
	}

	inline virtual T backward(T val) const
	{
		T t = tanh((double)val);
		return (T)1 - t * t;
	}
};

};

#endif
