#if !defined(ENN_DENSE_LAYER_H)
#define ENN_DENSE_LAYER_H

#include <LayerBase.h>
#include <ReLUActivation.h>
#include <SigmoidActivation.h>
#include <ProgmemHelper.h>

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
		_weights = (T*)malloc(sizeof(T) * (num_inputs + (BIAS?1:0)) * num_outputs);
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
		this->inputs((T*)malloc(sizeof(T) * num_inputs));
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		_weights = (T*)malloc(sizeof(T) * (input.num_inputs() + (BIAS?1:0)) * num_outputs);
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, T * weights, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		_weights = weights;
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, const ProgmemHelper<T> & weights, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		_weights = weights.read((input.num_inputs() + (BIAS?1:0)) * num_outputs);
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}

	///
	/// returns a pointer to errors
	///
	virtual inline const T* errors() const { return _errors; }
	virtual inline T* errors() { return _errors; }
	virtual inline void errors(T * errors)  { _errors = errors; }
	virtual inline T_SIZE num_errors() const { return _num_inputs; }


	///
	/// returns a pointer to weights
	///
	virtual inline const T* weights() const { return _weights; };
	virtual inline T* weights() { return _weights; };
	virtual inline void weights(T * weights) { _weights = weights; };
	virtual inline T_SIZE num_weights() const { return (_num_inputs + (BIAS?1:0)) * _num_outputs; }

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
		T acc;
		T_SIZE i, j;
		const T * in_p;
		const T * w_p;
		T * out_p;

		// iterate over outputs
		for (j = 0, out_p = _outputs, w_p = _weights; j < _num_outputs; j++) {
			acc = 0;
			// iterate over inputs
			for (i = 0, in_p = _inputs; i < _num_inputs; i++) {
				acc += (*in_p) * (*w_p);

				++in_p;
				++w_p;
			}
			if (BIAS) {
				acc += (*w_p);
				++w_p;
			}
			*out_p = _activation.forward(acc);
			++out_p;
		}
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	/// errors are from previous layer for each output.
	///  will calculate errors for the next layer
	/// size of errors should be the same as number of output errors
	virtual void backwards(T * errors)
	{
		T * i_p = _errors;
		T * j_p;
		T * w_p;
		T delta;
		// apply activation derivative
		_activation.apply_backwards_inplace(errors, _num_outputs);

		// iterate over inputs
		for (T_SIZE i = 0; i < _num_inputs; i++) {
			w_p = _weights + i;
			j_p = errors;
			delta = 0;
			// iterate over outputs
			for (T_SIZE j = 0; j < _num_outputs; j++) {
				delta += *j_p * *w_p;
				++j_p;
				w_p += _num_inputs;
			}
			*i_p = delta;
			++i_p;
		}
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
