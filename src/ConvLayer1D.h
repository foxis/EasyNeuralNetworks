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
	T_SIZE _stride;
	#define ENN_CONV_NEURON_SIZE(N_IN, K, STRIDE) ( ((N_IN) - (K)) / (STRIDE) + 1)
	#define ENN_CONV_OUTPUT_SIZE(N_IN, K, NK, STRIDE) (ENN_CONV_NEURON_SIZE(N_IN, K, STRIDE) * NK)
public:
	ConvLayer1D(T_SIZE num_inputs, T_SIZE num_kernels, T_SIZE kernel_size, T_SIZE stride, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(NULL, num_inputs, NULL, ENN_CONV_OUTPUT_SIZE(num_inputs, kernel_size, num_kernels, stride), activation) {
		_kernel_size = kernel_size;
		_num_kernels = num_kernels;
		this->_stride = stride;
		this->weights((T*)malloc(sizeof(T) * this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * this->num_outputs()));
		this->inputs((T*)malloc(sizeof(T) * num_inputs));
	}
	ConvLayer1D(LayerBase<T, BIAS, T_SIZE> &input, T_SIZE num_kernels, T_SIZE kernel_size, T_SIZE stride, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, ENN_CONV_OUTPUT_SIZE(input.num_inputs(), kernel_size, num_kernels, stride), activation) {
		_kernel_size = kernel_size;
		_num_kernels = num_kernels;
		this->_stride = stride;
		this->weights((T*)malloc(sizeof(T) * this->num_weights()));
		this->outputs((T*)malloc(sizeof(T) * this->num_outputs()));
	}
	ConvLayer1D(LayerBase<T, BIAS, T_SIZE> &input, T * weights, T_SIZE num_kernels, T_SIZE kernel_size, T_SIZE stride, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, ENN_CONV_OUTPUT_SIZE(input.num_inputs(), kernel_size, num_kernels, stride), activation) {
		_kernel_size = kernel_size;
		_num_kernels = num_kernels;
		this->_stride = stride;
		this->weights(weights);
		this->outputs((T*)malloc(sizeof(T) * this->num_outputs()));
	}
	ConvLayer1D(LayerBase<T, BIAS, T_SIZE> &input, const ProgmemHelper<T> & weights, T_SIZE num_kernels, T_SIZE kernel_size, T_SIZE stride, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, ENN_CONV_OUTPUT_SIZE(input.num_inputs(), kernel_size, num_kernels, stride), activation) {
		_kernel_size = kernel_size;
		_num_kernels = num_kernels;
		this->_stride = stride;
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
	virtual inline T_SIZE num_weights() const { return (this->_kernel_size + ENN_BIAS) * this->_num_kernels; }

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
		T * out = this->outputs();
		T * W = _weights;
		const T_SIZE neuron_size = this->num_inputs() / _stride - _kernel_size;
		for (T_SIZE i = 0; i < _num_kernels; i ++) {
			convolve_1d<T, BIAS, T_SIZE, false>(out, this->inputs(), W, this->num_inputs(), neuron_size, _stride);
			out += neuron_size;
			W += _kernel_size + ENN_BIAS;
		}
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
		// apply activation derivative
		this->_activation.apply_backward_inplace(deltas, this->outputs(), this->num_outputs());

		T * D = deltas;
		T * W = _weights;
		const T_SIZE neuron_size = this->num_inputs() / _stride - _kernel_size;

		memset(_errors, sizeof(T) * this->num_inputs());

		for (T_SIZE i = 0; i < _num_kernels; i ++) {
			convolve_1d<T, BIAS, T_SIZE, true>(_errors, D, W, neuron_size, this->num_inputs(), _stride);
			W += _kernel_size + ENN_BIAS;
			D += neuron_size;
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
