#if !defined(ENN_CONV_LAYER_1D_H)
#define ENN_CONV_LAYER_1D_H

#include <core/LayerBase.h>

namespace EasyNeuralNetworks {

///
/// This layer performs 1D convolution over the input of size (N, 1, M),
/// where N is the number of time-samples and M number of channels
///
/// Weights are organized as follows:
/// Wijk = W[i + j * N + k * (N * M + 1)], i < N, j < M, k < K
/// 		where:
///				N is the kernel width
///				M is the number of input channels
///				K is the number of kernels
///				k is the kernel number
///				j is the input channel number
///
/// weights shape is (N * M + 1, 1, K), where +1 is reserved for bias
/// Basically weights tensor contains embedded tensors inside for each kernel,
/// where the embedded kernel is stored as a tensor of shape (N, 1, M) + bias
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
	ConvLayer1D(T_INPUT& input, T_SIZE kernel_width, T_SIZE num_kernels, T_SIZE stride, T_INPUT& weights, const T_ACTIVATION& activation)
		: ConvLayer1D(input, kernel_width, num_kernels, stride, activation) {
		assert(weights.size() == this->weights().size());
		this->weights(weights);
	}

	ConvLayer1D(T_INPUT& input, T_SIZE kernel_width, T_SIZE num_kernels, T_SIZE stride, const T_ACTIVATION& activation)
		: T_LAYER(input, activation) {
		assert(input.height() == 1);
		_stride = stride;
		_kernel_width = kernel_width;
		this->outputs().resize((input.width() - kernel_width) / stride + 1, 1, num_kernels);
		this->weights().resize(kernel_width * input.depth() + ENN_BIAS, 1, num_kernels);
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
				T * W = kernel.data() + channel * _kernel_width;
				convolve_1d_add<T, T_SIZE, false>(feature_map, this->inputs().data(channel), W, this->inputs().width(), _kernel_width, _stride);
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
				convolve_1d_add<T, T_SIZE, true>(gradients().data(channel), G, W, this->inputs().width(), _kernel_width, _stride);
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
