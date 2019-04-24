#if !defined(ENN_INPUT_LAYER_H)
#define ENN_INPUT_LAYER_H

#include <LayerBase.h>
#include <LUActivation.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class InputLayer : public LayerBase<T, BIAS, T_SIZE> {
public:
	InputLayer(T_SIZE num_inputs) : LayerBase<T, BIAS, T_SIZE>(NULL, num_inputs, NULL, num_inputs, LUActivation<T>()) {
		T * p = (T*)malloc(sizeof(T) * num_inputs);
		this->inputs(p);
		this->outputs(p);
	}

	InputLayer(T * p, T_SIZE num_inputs) : LayerBase<T, BIAS, T_SIZE>(p, num_inputs, p, num_inputs, LUActivation<T>()) {
	}

	///
	/// returns a pointer to errors
	///
	virtual inline T* errors() { return NULL; }

	///
	/// returns a pointer to weights
	///
	virtual inline const T* weights() const { return NULL; };
	virtual inline T* weights() { return NULL; };
	virtual inline void weights(T * weights) { };

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward()
	{
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	virtual void backwards(const T * errors)
	{
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update()
	{
	}
};

};

#endif
