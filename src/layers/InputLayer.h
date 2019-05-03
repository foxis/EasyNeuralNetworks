#if !defined(ENN_INPUT_LAYER_H)
#define ENN_INPUT_LAYER_H

#include <core/LayerBase.h>
#include <activations/LUActivation.h>


namespace EasyNeuralNetworks {

///
/// This is the input layer. Must be the very first layer in the network.
/// NOTE: This layer may be omitted from the NN altogether.
///
template <typename T = ENN_DEFAULT_TYPE,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class InputLayer : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
public:
	InputLayer(T_SIZE width, T_SIZE height = 1, T_SIZE depth = 1) : T_LAYER(LUActivation<T, T_SIZE>()) {
		this->inputs().resize(width, height, depth);
		this->outputs(this->inputs());
	}

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward() { }

	virtual void training_begin() {}
	virtual void training_end() {}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	virtual void backward(T_INPUT& gradients) { }

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T_INPUT& gradients, T alpha) { }
};

};

#endif
