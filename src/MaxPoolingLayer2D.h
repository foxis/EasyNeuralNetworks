#if !defined(ENN_MAX_POOLING_LAYER_2D_H)
#define ENN_MAX_POOLING_LAYER_2D_H

#include <LayerBase.h>
#include <LUActivation.h>
#include <stdlib.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class MaxPoolingLayer2D : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T_SIZE _kernel_width;
	T_SIZE _kernel_height;
	bool training = false;
public:
	MaxPoolingLayer2D(T_LAYER& input, T_SIZE width, T_SIZE height) : MaxPoolingLayer2D(input.inputs(), width, height) {}
	MaxPoolingLayer2D(T_INPUT& input, T_SIZE width, T_SIZE height) : T_LAYER(input, LUActivation<T>()) {
		assert(input.width() % width == 0);
		assert(input.height() % height == 0);
		assert(width > 1);
		assert(height > 1);
		this->outputs().resize(input.width() / width, input.height() / height, input.depth());
		_kernel_width = width;
		_kernel_height = height;
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
				T * I = this->inputs().data(m * _kernel_height, i);
				if (training) {
					T * W = this->weights().data(m * _kernel_height, i);
					for (T_SIZE n = 0; n < this->outputs().width(); n++) {
						T_SIZE x = 0, y = 0;
						*O = max_mat<T, T_SIZE>(&x, &y, I, this->inputs().width(), _kernel_width, _kernel_height);
						*(W + x + y * this->inputs().width()) = 1;
						I += _kernel_width;
						W += _kernel_width;
						++O;
					}
				} else {
					for (T_SIZE n = 0; n < this->outputs().width(); n++) {
						T_SIZE x = 0, y = 0;
						*O = max_mat<T, T_SIZE>(&x, &y, I, this->inputs().width(), _kernel_width, _kernel_height);
						I += _kernel_width;
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
