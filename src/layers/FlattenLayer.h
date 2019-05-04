#if !defined(ENN_FLATTEN_LAYER_H)
#define ENN_FLATTEN_LAYER_H

#include <core/LayerBase.h>
#include <activations/LUActivation.h>
#include <layers/ReshapeLayer.h>

namespace EasyNeuralNetworks {

///
/// This layer reshapes the input to (N, 1, 1), where N is the size of the input tensor
///
template <typename T = ENN_DEFAULT_TYPE,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class FlattenLayer : public ReshapeLayer<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T _dropout_percent;
public:
	FlattenLayer(T_INPUT& input) : ReshapeLayer<T, T_SIZE>(input, input.size(), 1, 1) {}

};


};

#endif
