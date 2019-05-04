#if !defined(ENN_CONV_LAYER_2D_H)
#define ENN_CONV_LAYER_2D_H

#include <core/LayerBase.h>

namespace EasyNeuralNetworks {

///
/// This layer performs 2D convolution over the input of size (N, M, K),
/// where NxM is the image width/height and K number of channels
///
/// Weights are organized as follows:
/// Wijmk = W[i + j * N + m * N * M + k * (N * M * C + 1)], i < N, j < M, m < C, k < K
/// 		where:
///				N, M is the kernel width and height
///				K is the number of kernels
///				C is the number of input channels
///				k is the kernel number
///				m is the input channel number
///
/// weights shape is (N * M * C + 1, 1, K), where +1 is reserved for bias
/// Basically weights tensor contains embedded tensors inside for each kernel,
/// where the embedded kernel is stored as a tensor of shape (N, M, C) + bias
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
	ConvLayer2D(T_LAYER& input, T_SIZE kernel_width, T_SIZE kernel_height, T_SIZE num_kernels, T_SIZE stride, T_LAYER& weights, const T_ACTIVATION& activation)
		: ConvLayer2D(input.inputs(), kernel_width, kernel_height, num_kernels, stride, weights, activation) { }

	ConvLayer2D(T_LAYER& input, T_SIZE kernel_width, T_SIZE kernel_height, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: ConvLayer2D(input.inputs(), kernel_width, kernel_height, num_kernels, stride, activation) { }

	ConvLayer2D(T_INPUT& input, T_SIZE kernel_width, T_SIZE kernel_height, T_SIZE num_kernels, T_SIZE stride, T_LAYER& weights, const T_ACTIVATION& activation)
		: ConvLayer2D(input, kernel_width, kernel_height, num_kernels, stride, activation) {
		assert(weights.size() == this->weights().size());
		this->weights(weights);
	}

	ConvLayer2D(T_INPUT& input, T_SIZE kernel_width, T_SIZE kernel_height, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: T_LAYER(input, activation) {
		_stride = stride;
		_kernel_width = kernel_width;
		_kernel_height = kernel_height;
		this->outputs().resize((input.width() - kernel_width) / stride + 1, (input.height() - kernel_height) / stride + 1, num_kernels);
		this->weights().resize(kernel_width * kernel_height * input.depth() + ENN_BIAS, 1, num_kernels);
	}

	///
	///
	///
	virtual void forward()
	{
		this->outputs().fill(0);
		for (T_SIZE i = 0; i < this->weights().depth(); i ++) {
			auto feature_map = this->outputs().window(i, 1);
			auto kernel = this->weights().window(i, 1);
			for (T_SIZE channel = 0; channel < this->inputs().depth(); channel++) {
				T * W = kernel.data() + channel * _kernel_width * _kernel_height;
				convolve_2d_add<T, T_SIZE, false>(feature_map, this->inputs().data(channel), W, this->inputs().width(), this->inputs().height(), _kernel_width, _kernel_height, _stride);
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

		this->gradients().fill(0);
		for (T_SIZE i = 0; i < this->weights().depth(); i ++) {
			auto kernel = this->weights().window(i, 1);
			auto G = gradients.data(i);
			for (T_SIZE channel = 0; channel < this->inputs().depth(); channel++) {
				T * W = kernel.data() + channel * _kernel_width;
				convolve_2d_add<T, T_SIZE, true>(gradients().data(channel), G, W, this->inputs().width(), this->inputs().height(), _kernel_width, _kernel_height, _stride);
			}
		}
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
