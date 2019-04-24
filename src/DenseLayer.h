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
		this->weights((T*)malloc(sizeof(T) * this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
		this->inputs((T*)malloc(sizeof(T) * num_inputs));
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		this->weights((T*)malloc(sizeof(T) * this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, T * weights, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		this->weights(weights);
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}
	DenseLayer(LayerBase<T, BIAS, T_SIZE> &input, const ProgmemHelper<T> & weights, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		this->weights(weights.read(this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}

	///
	/// returns a pointer to errors
	///
	virtual inline const T* errors() const { return _errors; }
	virtual inline T* errors() { return _errors; }
	virtual inline void errors(T * errors)  { _errors = errors; }
	virtual inline T_SIZE num_errors() const { return this->_num_inputs; }


	///
	/// returns a pointer to weights
	///
	virtual inline const T* weights() const { return _weights; };
	virtual inline T* weights() { return _weights; };
	virtual inline void weights(T * weights) { _weights = weights; };
	virtual inline T_SIZE num_weights() const { return (this->_num_inputs + (BIAS?1:0)) * this->_num_outputs; }

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
		for (j = 0, out_p = this->_outputs, w_p = _weights; j < this->_num_outputs; j++) {
			acc = 0;
			// iterate over inputs
			for (i = 0, in_p = this->_inputs; i < this->_num_inputs; i++) {
				acc += (*in_p) * (*w_p);

				++in_p;
				++w_p;
			}
			if (BIAS) {
				acc += (*w_p);
				++w_p;
			}
			*out_p = this->_activation.forward(acc);
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
	virtual void backwards(T * deltas)
	{
		T * i_p = _errors;
		T * d_p;
		T * w_p;
		T delta;
/** /
		Serial.print("t - o: ");
		for (int i = 0 ; i < this->_num_outputs; i ++) {
			Serial.print(deltas[i]); Serial.print(" ");
		}
		Serial.print("  in: ");
		for (int i = 0 ; i < this->_num_inputs; i ++) {
			Serial.print(this->inputs()[i]); Serial.print(" ");
		}
		Serial.print("  weights: ");
		for (int i = 0 ; i < this->num_weights(); i ++) {
			Serial.print(this->weights()[i]); Serial.print(" ");
		}
		Serial.print("  out: ");
		for (int i = 0 ; i < this->_num_outputs; i ++) {
			Serial.print(this->outputs()[i]); Serial.print(" ");
		}
		Serial.println();

		Serial.print("delta: ");
		for (int i = 0 ; i < this->_num_outputs; i ++) {
			Serial.print(deltas[i]); Serial.print(" ");
		}
		Serial.println();
/***/
		// apply activation derivative
		this->_activation.apply_backward_inplace(deltas, this->outputs(), this->_num_outputs);

		// iterate over inputs
		for (T_SIZE i = 0; i < this->_num_inputs; i++) {
			w_p = _weights + i;
			d_p = deltas;
			delta = 0;
			// iterate over outputs
			for (T_SIZE j = 0; j < this->_num_outputs; j++) {
				delta += *d_p * *w_p;
				++d_p;
				w_p += (this->_num_inputs + (BIAS?1:0));
			}
			*i_p = delta;
			++i_p;
		}
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T * deltas, T alpha)
	{
		T * i_p;
		const T * e_p = deltas;
		T * w_p = _weights;

//		Serial.print("change: ");

		// iterate over output errors
		for (T_SIZE j = 0; j < this->_num_outputs; j++) {
			// iterate over inputs
			i_p = this->_inputs;
			for (T_SIZE i = 0; i < this->_num_inputs; i++) {
//					Serial.print(alpha * *e_p * *i_p); Serial.print(" ");
					*w_p += alpha * *e_p * *i_p;
					++i_p;
					++w_p;
			}
			// update bias
			if (BIAS) {
				*w_p += alpha * *e_p;
				++w_p;
			}
			++e_p;
		}
//		Serial.println();
	}

};

};

#endif
