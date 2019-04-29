#if !defined(ENN_CONV_LAYER_1D_H)
#define ENN_CONV_LAYER_1D_H

#include <LayerBase.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class ConvLayer1D : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	T_SIZE _stride;
	T_SIZE _kernel_width;
public:
	ConvLayer1D(T_LAYER& input, T_INPUT& weights, T_SIZE kernel_width, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: ConvLayer1D(input.inputs(), weights, kernel_width, num_kernels, stride, activation) {}

	ConvLayer1D(T_LAYER& input, T_SIZE kernel_width, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: ConvLayer1D(input.inputs(), kernel_width, num_kernels, stride, activation) {}

	ConvLayer1D(T_INPUT& input, T_SIZE kernel_width, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: T_LAYER(input, activation) {
		_stride = stride;
		_kernel_width = kernel_width;
		this->outputs().resize((input.width() * input.height() - kernel_width) / stride + 1, 1, num_kernels);
		this->weights().resize(kernel_width * input.depth() + ENN_BIAS, 1, num_kernels);
	}

	ConvLayer1D(T_SIZE width, T_SIZE depth, T_SIZE kernel_width, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: T_LAYER(activation) {
		_stride = stride;
		_kernel_width = kernel_width;
		this->inputs().resize(width, 1, depth);
		this->outputs().resize((width - kernel_width) / stride + 1, 1, num_kernels);
		this->weights().resize(kernel_width * depth + ENN_BIAS, 1, num_kernels);
	}

	ConvLayer1D(T_INPUT& input, T_INPUT& weights, T_SIZE kernel_width, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: T_LAYER(input, activation) {
		assert(weights.width() == kernel_width * input.depth() + ENN_BIAS);
		assert(weights.height() == 1);
		assert(weights.depth() == num_kernels);
		_stride = stride;
		_kernel_width = kernel_width;
		this->outputs().resize((input.width() * input.height() - kernel_width) / stride + 1, 1, num_kernels);
		this->weights(weights);
	}

	ConvLayer1D(T_SIZE width, T_SIZE depth, T_INPUT& weights, T_SIZE kernel_width, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: T_LAYER(activation) {
		assert(weights.width() == kernel_width * depth + ENN_BIAS);
		assert(weights.height() == 1);
		assert(weights.depth() == num_kernels);
		_stride = stride;
		_kernel_width = kernel_width;
		this->inputs().resize(width, 1, depth);
		this->outputs().resize((width - kernel_width) / stride + 1, 1, num_kernels);
		this->weights(weights);
	}

	///
	///
	///
	virtual void forward()
	{
		T_SIZE input_size = this->inputs().width();

		this->outputs().fill(0);
		for (T_SIZE i = 0; i < this->weights().depth(); i ++) {
			auto feature_map = this->outputs().window(i, 1);
			auto kernel = this->weights().window(i, 1);
			for (T_SIZE channel = 0; channel < this->inputs().depth(); channel++) {
				T * W = kernel.data() + channel * _kernel_width;
				convolve_1d_add<T, false, T_SIZE, false>(feature_map, this->inputs(), W, input_size, _kernel_width, _stride);
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
	///
	///
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
