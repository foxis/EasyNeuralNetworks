#if !defined(ENN_MAX_POOLING_LAYER_2D_H)
#define ENN_MAX_POOLING_LAYER_2D_H

#include <core/LayerBase.h>
#include <activations/LUActivation.h>

#include <stdlib.h>

namespace EasyNeuralNetworks {

///
/// This layer performs 2D max pooling with specified width, height and stride.
/// Default stride is equal to min(width, height).
/// Accepts any shape, but will perform max pooling along width and height axis
///
template <typename T = ENN_DEFAULT_TYPE,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class MaxPoolingLayer2D : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T_SIZE _kernel_width;
	T_SIZE _kernel_height;
	T_SIZE _stride;
	bool training = false;
public:
	MaxPoolingLayer2D(T_INPUT& input, T_SIZE width, T_SIZE height, T_SIZE stride=0) : T_LAYER(input, LUActivation<T>()) {
		assert(width > 1);
		assert(height > 1);
		if (stride == 0)
			stride = min(width, height);
		assert((input.width() - width) % stride == 0);
		assert((input.height() - height) % stride == 0);
		_kernel_width = width;
		_kernel_height = height;
		_stride = stride;
		this->outputs().resize((input.width() - width) / stride + 1, (input.height() - height) / stride + 1, input.depth());
	}

	///
	///
	///
	virtual void forward()
	{
		T * O = this->outputs();
		T_SIZE width = this->inputs().width();
		this->weights().fill(0);
		for (T_SIZE i = 0; i < this->inputs().depth(); i++) {
			for (T_SIZE m = 0; m < this->outputs().height(); m++) {
				T * I = this->inputs().data(m * _stride, i);
				if (training) {
					T * W = this->weights().data(m * _stride, i);
					for (T_SIZE n = 0; n < this->outputs().width(); n++) {
						T_SIZE x = 0, y = 0;
						*O = max_mat<T, T_SIZE>(&x, &y, I, this->inputs().width(), _kernel_width, _kernel_height);
						*(W + x + y * this->inputs().width()) = 1;
						I += _stride;
						W += _stride;
						++O;
					}
				} else {
					for (T_SIZE n = 0; n < this->outputs().width(); n++) {
						T_SIZE x = 0, y = 0;
						*O = max_mat<T, T_SIZE>(&x, &y, I, this->inputs().width(), _kernel_width, _kernel_height);
						I += _stride;
						++O;
					}
				}
			}
		}
	}

	virtual void training_begin()
	{
		this->gradients().resize(this->inputs());
		this->widths().resize(this->inputs());
	}
	virtual void training_end()
	{
		this->gradients().resize(0, 0, 0);
		this->widths().resize(0, 0, 0);
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	virtual void backward(T_INPUT& gradients) {
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T_INPUT& gradients, T alpha) {
	}
};

};

#endif
