#if !defined(ENN_CONV_LAYER_1D_H)
#define ENN_CONV_LAYER_1D_H

#include <LayerBase.h>
#include <ProgmemHelper.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class ConvLayer1D : public LayerBase<T, BIAS, T_SIZE> {
	T * _weights;
	T * _errors;
	T_SIZE _kernel_size;
	T_SIZE _num_kernels;
public:
	ConvLayer1D(T_SIZE num_inputs, T_SIZE num_kernels, T_SIZE kernel_size, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(NULL, num_inputs, NULL, (num_inputs - kernel_size) * num_kernels, activation) {
		_kernel_size = kernel_size;
		_num_kernels = num_kernels;
		this->weights((T*)malloc(sizeof(T) * this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * this->num_outputs()));
		this->inputs((T*)malloc(sizeof(T) * num_inputs));
	}
	ConvLayer1D(LayerBase<T, BIAS, T_SIZE> &input, T_SIZE num_kernels, T_SIZE kernel_size, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, (input.num_inputs() - kernel_size) * num_kernels, activation) {
		_kernel_size = kernel_size;
		_num_kernels = num_kernels;
		this->weights((T*)malloc(sizeof(T) * this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * this->num_outputs()));
	}
	ConvLayer1D(LayerBase<T, BIAS, T_SIZE> &input, T * weights, T_SIZE num_kernels, T_SIZE kernel_size, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, (input.num_inputs() - kernel_size) * num_kernels, activation) {
		_kernel_size = kernel_size;
		_num_kernels = num_kernels;
		this->weights(weights);
		this->outputs((T*)malloc(sizeof(T) * this->num_outputs()));
	}
	ConvLayer1D(LayerBase<T, BIAS, T_SIZE> &input, const ProgmemHelper<T> & weights, T_SIZE num_kernels, T_SIZE kernel_size, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, (input.num_inputs() - kernel_size) * num_kernels, activation) {
		_kernel_size = kernel_size;
		_num_kernels = num_kernels;
		this->weights(weights.read(this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * this->num_outputs()));
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
	virtual inline T_SIZE num_weights() const { return (this->_kernel_size + (BIAS?1:0)) * this->_num_kernels; }

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	///
	/// will calculate the output for 1d convolution NN
	///  kernel weight: i < kernel_size
	///  kernel:        j < num_kernels
	///	 kernel in pos: t < num_inputs - kernel_size
	///
	/// Otj = SUMi I(i+t) * Wij
	///
	///  Wkj = W_arr(k + j * kernel_size)
	virtual void forward()
	{
		T acc;
		T_SIZE i, j, t;
		T_SIZE stride_num = this->num_inputs() - _kernel_size; // valid padding
		const T * in_p;
		const T * w_p;
		T * out_p;

		// iterate over kernels
		for (j = 0, out_p = this->outputs(); j < _num_kernels; j++) {
			// iterate over kernel positions
			for (t = 0; t < stride_num; t++) {
				in_p = this->inputs() + t;
				w_p = _weights + j * (_kernel_size + (BIAS?1:0)) * stride_num;
				acc = 0;
				// iterate over kernel weights
				for (i = 0; i < _kernel_size; i++) {
					acc += *in_p * *w_p;
					++w_p;
					++in_p;
				}
				if (BIAS)
					acc += *w_p;
				*out_p = this->_activation.forward(acc);
				++out_p;
			}
		}
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
		T_SIZE i, j, t;
		T_SIZE stride_num = this->num_inputs() - _kernel_size; // valid padding
		const T * e_p;
		const T * w_p;
		T * d_p;

		// apply activation derivative
		this->_activation.apply_backward_inplace(deltas, this->outputs(), this->num_outputs());

		memset(_errors, sizeof(T) * this->num_inputs());

		// iterate over kernels
		for (j = 0, d_p = deltas; j < _num_kernels; j++) {
			// iterate over kernel positions
			for (t = 0; t < stride_num; t++) {
				e_p = _errors + t;
				w_p = _weights + j * (_kernel_size + ENN_BIAS) * stride_num;
				// iterate over kernel weights
				for (i = 0; i < _kernel_size; i++) {
					*e_p += *w_p * *d_p;
					++w_p;
					++e_p;
				}
				++d_p;
			}
		}
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
