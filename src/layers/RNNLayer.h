#if !defined(ENN_RNN_LAYER_H)
#define ENN_RNN_LAYER_H

#include <core/LayerBase.h>
#include <core/matvecop.h>
#include <activations/LUActivation.h>
#include "DenseLayer.h"

namespace EasyNeuralNetworks {

///
/// This layer is a fully connected Recurrent layer.
/// Can accept any shape of input. Output can be any shape.
///
/// This layer will flatten the input for computation.
/// Weights are organized as the same way as for a DenseLayer.
/// Note that recurrent weights are without bias.
template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class RNNLayer : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T_INPUT _recurrent_weights;
	T_INPUT _memory;
public:
	RNNLayer(T_INPUT& input, T_SIZE out_width, T_INPUT& weights, T_INPUT& recurrent_weights, const T_ACTIVATION& activation)
		: RNNLayer(input, out_width, 1, weights, recurrent_weights, activation) { }

	RNNLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_INPUT& weights, T_INPUT& recurrent_weights, const T_ACTIVATION& activation)
		: RNNLayer(input, out_width, out_height, 1, weights, recurrent_weights, activation) { }

	RNNLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_SIZE out_depth, T_INPUT& weights, T_INPUT& recurrent_weights, const T_ACTIVATION& activation)
		: RNNLayer(input, out_width, out_height, out_depth, activation) {
		assert(weights.size() == this->weights().size());
		assert(recurrent_weights.size() == _recurrent_weights.size());
		this->weights(weights);
		_recurrent_weights = recurrent_weights;
	}

	RNNLayer(T_INPUT& input, T_SIZE out_width, const T_ACTIVATION& activation)
		: RNNLayer(input, out_width, 1, activation) {	}

	RNNLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, const T_ACTIVATION& activation)
		: RNNLayer(input, out_width, out_height, 1, activation) { }

	RNNLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_SIZE out_depth, const T_ACTIVATION& activation)
		: T_LAYER(input, activation)
	{
		this->outputs().resize(out_width, out_height, out_depth);
		_memory.resize(this->outputs());
		this->weights().resize(input.size() + ENN_BIAS, this->outputs().size(), 1);
		this->_recurrent_weights.resize(this->outputs.size(), this->outputs.size(), 1);
	}

	///
	///
	///
	virtual void forward()
	{
		mat_mul<T, false, T_SIZE, false>(_memory, this->outputs(), _recurrent_weights, this->outputs().size(), this->outputs().size());
		mat_mul<T, BIAS, T_SIZE, false>(this->outputs(), this->inputs(), this->weights(), this->inputs().size(), this->outputs().size());
		sum_arr<T, T_SIZE>(this->outputs(), this->outputs(), _memory, this->outputs().size());
		this->activation().apply_forward_inplace(this->outputs());
	}

	virtual void training_begin() {
	}
	virtual void training_end() {
	}

	///
	///
	///
	virtual void backward(T_INPUT& gradients)
	{
	}

	///
	///
	///
	virtual void update(const T_INPUT& gradients, T alpha)
	{
	}
};

};

#endif
