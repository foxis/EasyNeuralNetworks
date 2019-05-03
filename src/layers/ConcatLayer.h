#if !defined(ENN_CONCAT_LAYER_H)
#define ENN_CONCAT_LAYER_H

#include <core/LayerBase.h>
#include <activations/LUActivation.h>
#include <vector>
#include <cstdarg>

namespace EasyNeuralNetworks {

///
/// This layer concatenates a collection of layers/tensors into one large output
/// concatenation is done depth-wise, so each input layer should have the same width and height
/// NOTE: Due to current NeuralNetwork implementation this layer prevents training by design
/// NOTE: Be aware, that it accepts a collection of pointers to either a layer or tensor.
///       This is indicated by IS_LAYER. Memory corruption may happen if IS_LAYER==true, but tensors are passed.
///
/// example(inception net block):
/// InputLayer<float> 				layer0(...);
/// ConvLayer2D<float> 				layer1(layer0...);
/// ConvLayer2D<float> 				layer2(layer0...);
/// MaxPoolingLayer2D<float> 	layer3(layer0, ...);
/// ConcatLayer<float, true> 	output(3, &layer1, &layer2, &layer3);
///
/// NeuralNetwork<float>			nn(5, &layer0, &layer1, &layer2, &layer3, &output);
template <typename T,
					bool IS_LAYER = true,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class ConcatLayer : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	std::vector<T_INPUT> _inputs;
public:
	ConcatLayer(T_SIZE num_layers, ...) : T_LAYER(LUActivation<T>()) {
		va_list layers;
		va_start(layers, num_layers);
		for (T_SIZE i = 0; i < num_layers; i++) {
			if (IS_LAYER)
				_inputs.push_back(va_arg( layers, T_LAYER* )->outputs());
			else
				_inputs.push_back(va_arg( layers, T_INPUT* ));
		}
		va_end(layers);

		T_SIZE width, height, depth, index = 0;
		for (auto &I : _inputs) {
			if (index == 0) {
				width = I.width();
				height = I.height();
				depth = I.depth();
			} else {
				assert(width == I.width() && height == I.height());
				depth += I.depth();
			}
			++index;
		}

		this->outputs().resize(width, height, depth);
	}

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward()
	{
		T_SIZE depth = 0;
		for (auto &I : _inputs) {
			T_INPUT& dst = this->outputs().window(depth, I.depth());
			dst.copy(I.outputs());
			depth += I.depth();
		}
	}

	virtual void training_begin()
	{
		assert(false);
	}
	virtual void training_end()
	{
	}

	///
	///
	virtual void backward(T_INPUT& gradients)
	{
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T_INPUT& gradients, T alpha) { }
};

};

#endif
