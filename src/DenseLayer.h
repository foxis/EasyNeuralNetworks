#if !defined(ENN_DENSE_LAYER_H)
#define ENN_DENSE_LAYER_H

#include <LayerBase.h>
#include <ProgmemHelper.h>
#include <matvecop.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class DenseLayer : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
public:
	DenseLayer(T_LAYER& input, T_LAYER& output, const T_ACTIVATION& activation)
		: DenseLayer(input.outputs(), output.inputs(), activation) {}

	DenseLayer(T_LAYER& input, T_LAYER& output, T_INPUT& weights, const T_ACTIVATION& activation)
			: T_LAYER(input.outputs(), output.inputs(), weights, activation) {
		assert(weights.size() == (input.outputs().size() + ENN_BIAS) * output.inputs().size());
	}

	DenseLayer(T_LAYER& input, T_SIZE out_width, T_SIZE out_height, const T_ACTIVATION& activation)
		: DenseLayer(input.outputs(), out_width, out_height, activation) {}

	DenseLayer(T_LAYER& input, T_SIZE num_outputs, const T_ACTIVATION& activation)
		: DenseLayer(input.outputs(), num_outputs, activation) {}

	DenseLayer(T_LAYER& input, T_SIZE num_outputs, T_INPUT& weights, const T_ACTIVATION& activation)
		: T_LAYER(input.outputs(), activation) {
		assert(weights.size() == (input.outputs().size() + ENN_BIAS) * num_outputs);
		this->outputs().resize(num_outputs, 1, 1);
		this->weights(weights);
	}

	DenseLayer(T_LAYER& input, T_SIZE out_width, T_SIZE out_height, T_INPUT& weights, const T_ACTIVATION& activation)
		: T_LAYER(input.outputs(), activation) {
		assert(weights.size() == (input.outputs().size() + ENN_BIAS) * out_width * out_height);
		this->outputs().resize(out_width, out_height, 1);
		this->weights(weights);
	}

	DenseLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, const T_ACTIVATION& activation)
		: T_LAYER(input, activation)
	{
		this->outputs().resize(out_width, out_height, 1);
		this->weights().resize(input.size() + ENN_BIAS, this->outputs().size(), 1);
	}

	DenseLayer(T_INPUT& input, T_SIZE num_outputs, const T_ACTIVATION& activation)
		: DenseLayer(input, num_outputs, 1, activation) {	}

	DenseLayer(T_INPUT& input, T_INPUT& output, const T_ACTIVATION& activation)
		: T_LAYER(input, output, activation)
	{
		this->weights().resize(input.size() + ENN_BIAS, this->outputs().size(), 1);
	}

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	///
	/// will calculate the output for fully connected NN
	///
	/// Oj = SUMi Ii * Wij
	///
	///  Wij = W_arr(i + j * N)
	virtual void forward()
	{
		mat_mul<T, BIAS, T_SIZE, false>(this->outputs(), this->inputs(), this->weights(), this->inputs().size(), this->outputs().size());
		this->_activation.apply_forward_inplace(this->outputs());
	}

	virtual void training_begin() {
		this->gradients().resize(this->inputs());
	}
	virtual void training_end() {
		this->gradients().resize(0, 0, 0);
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	/// errors are from previous layer for each output.
	///  will calculate errors for the next layer
	/// size of errors should be the same as number of output errors
	virtual void backward(T_INPUT& gradients)
	{
		// calculate gradients
		this->_activation.apply_backward_inplace(gradients, this->outputs());
		mat_mul<T, BIAS, T_SIZE, true>(this->gradients(), gradients, this->weights(), this->inputs().size(), this->outputs().size());
	}

	///
	/// will update the weights calculated in backward
	///
	virtual void update(const T_INPUT& gradients, T alpha)
	{
		outer_product_add_const<T, BIAS, T_SIZE>(this->weights(), gradients, this->inputs(), this->outputs().size(), this->inputs().size(), -alpha);
	}

};

};

#endif
