#if !defined(ENN_ZERO_PADDING_LAYER_2D_H)
#define ENN_ZERO_PADDING_LAYER_2D_H

#include <core/LayerBase.h>
#include <activations/LUActivation.h>

namespace EasyNeuralNetworks {

///
/// This layer adds padding of zeroes for width and height dimensions
///
template <typename T = ENN_DEFAULT_TYPE,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class ZeroPaddingLayer2D : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T_SIZE _padding_w;
	T_SIZE _padding_h;
public:
	ZeroPaddingLayer2D(T_INPUT& input, T_SIZE padding_w, T_SIZE padding_h) : T_LAYER(input, LUActivation<T>()) {
		this->outputs().resize(input.width() + 2 * padding_w, input.height() + 2 * padding_h, input.depth());
		this->outputs().fill(0);
		_padding_w = padding_w;
		_padding_h = padding_h;
	}

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward()
	{
		for (T_SIZE i = 0; i < this->inputs().depth(); i++) {
			T * src = this->inputs().data(i);
			T * dst = this->inputs().data(i) + _padding_w + _padding_h * this->inputs().width();
			for (T_SIZE i = 0; i < this->inputs().height(); i++) {
				memcpy(dst, src, sizeof(T) * this->inputs().width());
				dst += this->outputs().width();
				src += this->inputs().width();
			}
		}
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
	virtual void backward(T_INPUT& gradients)
	{
		for (T_SIZE i = 0; i < this->gradients().depth(); i++) {
			T * src = gradients.data(i) + _padding_w + _padding_h * this->gradients().width();
			T * dst = this->gradients().data(i);
			for (T_SIZE i = 0; i < this->gradients().height(); i++) {
				memcpy(dst, src, sizeof(T) * this->gradients().width());
				dst += this->gradients().width();
				src += gradients.width();
			}
		}
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T_INPUT& gradients, T alpha) { }
};

};

#endif
