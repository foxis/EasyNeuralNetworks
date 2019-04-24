#if !defined(ENN_LU_ACTIVATION_H)
#define ENN_LU_ACTIVATION_H

#include <LayerBase.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE>
class LUActivation : public ActivationBase<T> {
public:
	inline virtual T forward(T val) const
	{
		return val;
	}

	inline virtual T backward(T val) const
	{
		return 1;
	}
};

};

#endif
