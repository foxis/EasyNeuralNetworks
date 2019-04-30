#if !defined(ENN_SOFTMAX_ACTIVATION_H)
#define ENN_SOFTMAX_ACTIVATION_H

#include <core/LayerBase.h>
#include <math.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class SoftmaxActivation : public ActivationBase<T, T_SIZE> {
public:
	inline virtual T forward(T val) const
	{
		return exp((double)val);
	}

	inline virtual T backward(T val) const
	{
		return val * ((T)1 - val);
	}

	virtual void apply_forward_inplace(tensor<T, T_SIZE>& a) const {
		auto I = a.data();
		auto num = a.size();

		T mv = max_arr(NULL, I, num);
		T acc = 0;

		while (num--) {
			T tmp = forward(*I - mv);
			*I = tmp;
			acc += tmp;
			++I;
		}

		I = a.data();
		num = a.size();

		while (num--) {
			*I = *I / acc;
			++I;
		}
	}
};

};

#endif
