#if !defined(ENN_DENSE_LAYER_H)
#define ENN_DENSE_LAYER_H

#include <core/LayerBase.h>
#include <core/ProgmemHelper.h>
#include <core/matvecop.h>

namespace EasyNeuralNetworks {

///
/// This layer is a fully connected layer.
/// Can accept any shape of input. Output can be any shape.
///
/// This layer will flatten the input for computation.
/// Weights are organized as follows:
/// Wij = W[i + j * (N + 1)], i < N, j < M,
///     where N is the input size and M is the output size,
///           i = N is the bias
/// Weights shape is (N + 1, M, 1), where +1 reserved for biases
template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class DenseLayer : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
public:
	DenseLayer(T_INPUT& input, T_SIZE out_width, T_INPUT& weights, const T_ACTIVATION& activation)
		: DenseLayer(input, out_width, 1, weights, activation) { }

	DenseLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_INPUT& weights, const T_ACTIVATION& activation)
		: DenseLayer(input, out_width, out_height, 1, weights, activation) { }

	DenseLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_SIZE out_depth, T_INPUT& weights, const T_ACTIVATION& activation)
		: DenseLayer(input, out_width, out_height, out_depth, activation) {
		assert(weights.size() == this->weights().size());
		this->weights(weights);
	}

	DenseLayer(T_INPUT& input, T_SIZE out_width, const T_ACTIVATION& activation)
		: DenseLayer(input, out_width, 1, activation) {	}

	DenseLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, const T_ACTIVATION& activation)
		: DenseLayer(input, out_width, out_height, 1, activation) { }

	DenseLayer(T_INPUT& input, T_SIZE out_width, T_SIZE out_height, T_SIZE out_depth, const T_ACTIVATION& activation)
		: T_LAYER(input, activation)
	{
		this->outputs().resize(out_width, out_height, out_depth);
		this->weights().resize(input.size() + ENN_BIAS, this->outputs().size(), 1);
	}

	///
	///
	///
	virtual void forward()
	{
		mat_mul<T, BIAS, T_SIZE, false>(this->outputs(), this->inputs(), this->weights(), this->inputs().size(), this->outputs().size());
		this->activation().apply_forward_inplace(this->outputs());
	}

	virtual void training_begin() {
		this->gradients().resize(this->inputs());
	}
	virtual void training_end() {
		this->gradients().resize(0, 0, 0);
	}

	///
	///
	///
	virtual void backward(T_INPUT& gradients)
	{
		// calculate gradients
		this->_activation.apply_backward_inplace(gradients, this->outputs());
		mat_mul<T, BIAS, T_SIZE, true>(this->gradients(), gradients, this->weights(), this->inputs().size(), this->outputs().size());
	}

	///
	///
	///
	virtual void update(const T_INPUT& gradients, T alpha)
	{
		outer_product_add_const<T, BIAS, T_SIZE>(this->weights(), gradients, this->inputs(), this->outputs().size(), this->inputs().size(), -alpha);
	}
};

};

#endif
