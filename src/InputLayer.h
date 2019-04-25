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
		this->inputs((T*)malloc(sizeof(T) * num_inputs));
		this->outputs(this->inputs());
	}

	InputLayer(T * p, T_SIZE num_inputs) : LayerBase<T, BIAS, T_SIZE>(p, num_inputs, p, num_inputs, LUActivation<T>()) {
	}

	///
	/// returns a pointer to errors
	///
	virtual inline const T* errors() const { return NULL; }
	virtual inline T* errors() { return NULL; }
	virtual inline void errors(T * errors)  { }
	virtual inline T_SIZE num_errors() const { return 0; }

	///
	/// returns a pointer to weights
	///
	virtual inline const T* weights() const { return NULL; };
	virtual inline T* weights() { return NULL; };
	virtual inline void weights(T * weights) { };
	virtual inline T_SIZE num_weights() const { return 0; }

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
	virtual void backward(T * deltas)
	{
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T * deltas, T alpha)
	{
	}
};

};

#endif
