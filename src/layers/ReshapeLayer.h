#if !defined(ENN_RESHAPE_LAYER_H)
#define ENN_RESHAPE_LAYER_H

#include <core/LayerBase.h>
#include <activations/LUActivation.h>

namespace EasyNeuralNetworks {

///
/// This layer reshapes the input to whatever output required.
/// NOTE: The output size should be <= the input size
///
template <typename T = ENN_DEFAULT_TYPE,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class ReshapeLayer : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T _dropout_percent;
public:
	ReshapeLayer(T_INPUT& input, T_SIZE width, T_SIZE height, T_SIZE depth) : T_LAYER(input, LUActivation<T>()) {
		this->outputs(this->inputs());
		this->outputs().reshape(width, height, depth);
	}

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward()
	{
	}

	virtual void training_begin()
	{
	}
	virtual void training_end()
	{
		this->gradients.resize(0, 0, 0);
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	virtual void backward(T_INPUT& gradients)
	{
		this->gradients(gradients);
		this->gradients.reshape(this->inputs());
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T_INPUT& gradients, T alpha) { }
};

};

#endif
