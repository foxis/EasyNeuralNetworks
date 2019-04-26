#if !defined(ENN_DENSE_LAYER_H)
#define ENN_DENSE_LAYER_H

#include <LayerBase.h>
#include <ProgmemHelper.h>
#include <matvecop.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class DenseLayer : public LayerBase<T, BIAS, T_SIZE> {
	T * _weights;
	T * _errors;
public:
	DenseLayer(T_SIZE num_inputs, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(NULL, num_inputs, NULL, num_outputs, activation) {
		this->weights((T*)malloc(sizeof(T) * this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
		this->inputs((T*)malloc(sizeof(T) * num_inputs));
		_errors = NULL;
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		this->weights((T*)malloc(sizeof(T) * this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
		_errors = NULL;
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, T * weights, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		this->weights(weights);
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
		_errors = NULL;
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, const ProgmemHelper<T> & weights, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		this->weights(weights.read(this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
		_errors = NULL;
	}

	///
	/// returns a pointer to errors
	///
	virtual inline const T* errors() const { return _errors; }
	virtual inline T* errors() { return _errors; }
	virtual inline void errors(T * errors)  { _errors = errors; }
	virtual inline T_SIZE num_errors() const { return this->num_inputs(); }


	///
	/// returns a pointer to weights
	///
	virtual inline const T* weights() const { return _weights; };
	virtual inline T* weights() { return _weights; };
	virtual inline void weights(T * weights) { _weights = weights; };
	virtual inline T_SIZE num_weights() const { return (this->num_inputs() + ENN_BIAS) * this->num_outputs(); }

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
		mat_mul<T, BIAS, T_SIZE, false>(this->outputs(), this->inputs(), _weights, this->num_inputs(), this->num_outputs());
		this->_activation.apply_forward_inplace(this->outputs(), this->num_outputs());
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	/// errors are from previous layer for each output.
	///  will calculate errors for the next layer
	/// size of errors should be the same as number of output errors
	virtual void backward(T * deltas)
	{
		// calculate gradients
		this->_activation.apply_backward_inplace(deltas, this->outputs(), this->num_outputs());
		mat_mul<T, BIAS, T_SIZE, true>(this->errors(), deltas, _weights, this->num_inputs(), this->num_outputs());
	}

	///
	/// will update the weights calculated in backward
	///
	virtual void update(const T * deltas, T alpha)
	{
		outer_product_add_const<T, BIAS, T_SIZE>(_weights, deltas, this->inputs(), this->num_outputs(), this->num_inputs(), -alpha);
	}

};

};

#endif
