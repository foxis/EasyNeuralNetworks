#if !defined(ENN_LSTM_LAYER_H)
#define ENN_LSTM_LAYER_H

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
/// Weights are organized the same way as for a DenseLayer, except that
/// depth is 4 to hold i, f, c and o blocks with respective biases.
/// Recurrent weights are organized in the same way as weights, except that
/// they don't contain a bias.
/// Keras LSTMCell was taken as implementation reference (implementation=2)
template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class LSTMLayer : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	const T_ACTIVATION& _recurrent_activation;
	T_INPUT _recurrent_weights;
	T_INPUT _carry;
	T_INPUT _z;
public:
	LSTMLayer(T_INPUT& input, T_SIZE out_width, T_INPUT& weights, T_INPUT& recurrent_weights, const T_ACTIVATION& activation, const T_ACTIVATION& recurrent_activation)
		: LSTMLayer(input, out_width, 1, weights, recurrent_weights, activation, recurrent_activation) { }

	LSTMLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_INPUT& weights, T_INPUT& recurrent_weights, const T_ACTIVATION& activation, const T_ACTIVATION& recurrent_activation)
		: LSTMLayer(input, out_width, out_height, 1, weights, recurrent_weights, activation, recurrent_activation) { }

	LSTMLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_SIZE out_depth, T_INPUT& weights, T_INPUT& recurrent_weights, const T_ACTIVATION& activation, const T_ACTIVATION& recurrent_activation)
		: LSTMLayer(input, out_width, out_height, out_depth, activation, recurrent_activation) {
		assert(weights.size() == this->weights().size());
		assert(recurrent_weights.size() == _recurrent_weights.size());
		this->weights(weights);
		this->_recurrent_weights = recurrent_weights;
	}

	LSTMLayer(T_INPUT& input, T_SIZE out_width, const T_ACTIVATION& activation, const T_ACTIVATION& recurrent_activation)
		: LSTMLayer(input, out_width, 1, activation, recurrent_activation) {	}

	LSTMLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, const T_ACTIVATION& activation, const T_ACTIVATION& recurrent_activation)
		: LSTMLayer(input, out_width, out_height, 1, activation, recurrent_activation) { }

	LSTMLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_SIZE out_depth, const T_ACTIVATION& activation, const T_ACTIVATION& recurrent_activation)
		: T_LAYER(input, activation),
			_recurrent_activation(recurrent_activation)
	{
		this->outputs().resize(out_width, out_height, out_depth);

		this->weights().resize(input.size() + ENN_BIAS, this->outputs().size(), 4);
		this->_recurrent_weights.resize(this->outputs().size(), this->outputs().size(), 4);

		this->_carry.resize(this->outputs());
		this->_z.resize(this->outputs() * 4);
	}

	///
	///
	///
	virtual void forward()
	{
		// Python snippet code (w/o dropout masking):
		// h_tm1 = states[0]  # previous memory state
		// c_tm1 = states[1]  # previous carry state
		// .....
		// 		z = K.dot(inputs, self.kernel)
		// 		z += K.dot(h_tm1, self.recurrent_kernel)
		// 		if self.use_bias:
		// 				z = K.bias_add(z, self.bias)
		//
		// 		z0 = z[:, :self.units]
		// 		z1 = z[:, self.units: 2 * self.units]
		// 		z2 = z[:, 2 * self.units: 3 * self.units]
		// 		z3 = z[:, 3 * self.units:]
		//
		// 		i = self.recurrent_activation(z0)
		// 		f = self.recurrent_activation(z1)
		// 		c = f * c_tm1 + i * self.activation(z2)
		// 		o = self.recurrent_activation(z3)
		//
		// h = o * self.activation(c)
		// return h, [h, c]  # output, states

		mat_mul<T, BIAS, T_SIZE, false>(_z, this->inputs(), this->weights(), this->inputs().size(), _z.size());
		mat_mul_add<T, false, T_SIZE, false>(_z, this->outputs(), _recurrent_weights, this->outputs().size(), _z.size());

		T_INPUT z0(_z.window(0, 1));
		T_INPUT z1(_z.window(1, 1));
		T_INPUT z2(_z.window(2, 1));
		T_INPUT z3(_z.window(3, 1));

		_recurrent_activation.apply_forward_inplace(z0);	// i
		_recurrent_activation.apply_forward_inplace(z1);	// f
		this->activation().apply_forward_inplace(z2);			// c
		_recurrent_activation.apply_forward_inplace(z3);	// o

		// c = f * c_tm1 + i * self.activation(z2)
		hadamard_product<T, T_SIZE>(_carry, z1, _carry, _carry.size());
		hadamard_product_add<T, T_SIZE>(_carry, z0, z2, _carry.size());

		// h = o * self.activation(c)
		this->outputs().copy(_carry);
		this->activation().apply_forward_inplace(this->outputs());
		hadamard_product<T, T_SIZE>(this->outputs(), this->outputs(), z3, z3.size());
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
