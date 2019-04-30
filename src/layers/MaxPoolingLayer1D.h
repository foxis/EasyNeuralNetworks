#if !defined(ENN_MAX_POOLING_LAYER_1D_H)
#define ENN_MAX_POOLING_LAYER_1D_H

#include <core/LayerBase.h>
#include <activations/LUActivation.h>

#include <stdlib.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class MaxPoolingLayer1D : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T_SIZE _kernel_width;
	bool training = false;
public:
	MaxPoolingLayer1D(T_LAYER& input, T_SIZE width) : MaxPoolingLayer1D(input.inputs(), width) {}
	MaxPoolingLayer1D(T_INPUT& input, T_SIZE width) : T_LAYER(input, LUActivation<T>()) {
		assert(input.width() % width == 0);
		assert(width > 1);
		this->outputs().resize(input.width() / width, input.height(), input.depth());
		_kernel_width = width;
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
			for (T_SIZE j = 0; j < this->inputs().height(); j++) {
				T * I = this->inputs().data(j, i);
				if (training) {
					T * W = this->weights().data(j, i);
					for (T_SIZE k = 0; k < this->outputs().width(); k++) {
						T_SIZE idx = 0;
						*O = max_arr<T, T_SIZE>(&idx, I, _kernel_width);
						*(W + idx) = 1;
						I += _kernel_width;
						W += _kernel_width;
						++O;
					}
				} else {
					for (T_SIZE k = 0; k < this->outputs().width(); k++) {
						T_SIZE idx = 0;
						*O = max_arr<T, T_SIZE>(&idx, I, _kernel_width);
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
		training = true;
	}
	virtual void training_end()
	{
		this->gradients().resize(0, 0, 0);
		this->widths().resize(0, 0, 0);
		training = false;
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
