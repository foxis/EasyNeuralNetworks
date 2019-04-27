#if !defined(ENN_LU_ACTIVATION_H)
#define ENN_LU_ACTIVATION_H

#include <LayerBase.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class LUActivation : public ActivationBase<T, T_SIZE> {
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
