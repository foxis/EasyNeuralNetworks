#if !defined(ENN_CONV_LAYER_2D_H)
#define ENN_CONV_LAYER_2D_H

#include <LayerBase.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class ConvLayer2D : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	T_SIZE _stride;
	T_SIZE _kernel_width;
	T_SIZE _kernel_height;
public:
	ConvLayer2D(T_LAYER& input, T_SIZE kernel_width, T_SIZE kernel_height, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: ConvLayer2D(input.inputs(), kernel_width, kernel_height, num_kernels, stride, activation) {}

	ConvLayer2D(T_INPUT& input, T_SIZE kernel_width, T_SIZE kernel_height, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: T_LAYER(input, activation) {
		_stride = stride;
		_kernel_width = kernel_width;
		_kernel_height = kernel_height;
		this->outputs().resize((input.width() - kernel_width) / stride + 1, (input.height() - kernel_height) + 1, num_kernels);
		this->weights().resize(kernel_width * kernel_height * input.depth() + ENN_BIAS, 1, num_kernels);
	}

	ConvLayer2D(T_SIZE width, T_SIZE height, T_SIZE depth, T_SIZE kernel_width, T_SIZE kernel_height, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: T_LAYER(activation) {
		_stride = stride;
		_kernel_width = kernel_width;
		_kernel_height = kernel_height;
		this->inputs().resize(width, height, depth);
		this->outputs().resize((input.width() - kernel_width) / stride + 1, (input.height() - kernel_height) + 1, num_kernels);
		this->weights().resize(kernel_width * kernel_height * input.depth() + ENN_BIAS, 1, num_kernels);
	}

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
		this->outputs().fill(0);
		for (T_SIZE i = 0; i < this->weights().depth(); i ++) {
			auto feature_map = this->outputs().window(i, 1);
			auto kernel = this->weights().window(i, 1);
			for (T_SIZE channel = 0; channel < this->inputs().depth(); channel++) {
				T * W = kernel.data() + channel * _kernel_size;
				convolve_1d_add<T, false, T_SIZE, false>(feature_map, this->inputs(), W, input_size, _kernel_size, _stride);
			}
			sum_arr_add<T, T_SIZE>(feature_map, kernel[kernel.size() - 1], feature_map.size());
		}
		this->_activation.apply_forward_inplace(this->outputs());
	}

	virtual void training_begin()
	{
		this->gradients().resize(this->inputs());
	}
	virtual void training_end()
	{
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
		// apply activation derivative
		this->_activation.apply_backward_inplace(gradients, this->outputs());

	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T_INPUT& gradients, T alpha)
	{
	}

	inline T zip_mul_sum(const T * u, const T * v, T_SIZE N, T_SIZE M, T_SIZE stride) {
		T acc = 0;
		while (N && M) {
			acc += *u * *v;
			N -= stride;
			--M;
			u -= stride;
			++v;
		}

		return acc;
	}
};

};

#endif
