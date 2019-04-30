#if !defined(ENN_NEURAL_NETWORK_H)
#define ENN_NEURAL_NETWORK_H

#include <activations/LUActivation.h>
#include <activations/ReLUActivation.h>
#include <activations/SigmoidActivation.h>
#include <activations/TanhActivation.h>
#include <activations/SoftplusActivation.h>
#include <layers/InputLayer.h>
#include <layers/DenseLayer.h>
#include <layers/DropOutLayer.h>
#include <layers/DropOutLayer1D.h>
#include <layers/DropOutLayer2D.h>
#include <layers/ConvLayer1D.h>
#include <layers/ConvLayer2D.h>
#include <layers/MaxPoolingLayer1D.h>
#include <layers/MaxPoolingLayer2D.h>

#include <core/FixedPointType.h>
#include <cstdarg>
#include <vector>
#include <stdlib.h>

namespace EasyNeuralNetworks {

#define ENN_T_LOSS_TYPEDEF(T_LOSS_NAME) typedef LossFunctionBase<T, T_SIZE> T_LOSS_NAME;

template<typename T, typename T_SIZE>
class LossFunctionBase {
public:
	virtual T operator () (tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& target, const tensor<T, T_SIZE>& output) const = 0;

	#define ENN_LOSS_LOOP(DELTA, ACC) \
		T acc = 0;	\
		assert(deltas.size() == target.size() && deltas.size() == output.size()); \
		auto num = output.size(); \
		auto D = deltas.begin(1); \
		auto Ta = target.begin(1); \
		auto O = output.begin(1); \
		while (num--) {	\
			*D = DELTA;	\
			acc += ACC;	\
			++Ta;	\
			++O;	\
			++D;	\
		}	\
		return acc;
};

template<typename T, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class L2Loss : public LossFunctionBase<T, T_SIZE> {
public:
	virtual T operator () (tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& target, const tensor<T, T_SIZE>& output) const {
		ENN_LOSS_LOOP(*O - *Ta, *D * *D)
	}
};

template<typename T, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class L1Loss : public LossFunctionBase<T, T_SIZE> {
public:
	virtual T operator () (tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& target, const tensor<T, T_SIZE>& output) const {
		ENN_LOSS_LOOP((*Ta < *O ? 1 : *Ta > *O ? -1 : 0), abs(*D))
	}
};

template<typename T, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class CrossEntropy : public LossFunctionBase<T, T_SIZE> {
public:
	virtual T operator () (tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& target, const tensor<T, T_SIZE>& output) const {
		ENN_LOSS_LOOP(*Ta * log(*O), -*D)
	}
};


template<typename T, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class NeuralNetwork;

template<typename T, typename T_SIZE>
class TrainerBase {
protected:
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T_INPUT inputs;
	T_INPUT outputs;
	std::vector<T_LAYER*> layers;
	NeuralNetwork<T, T_SIZE>* network;
	T_LAYER *first;
	T_LAYER *last;

public:
	TrainerBase()
	{	}

	virtual void init(const T_INPUT &inputs, const T_INPUT &outputs, NeuralNetwork<T, T_SIZE>* network) {
		this->layers = network->layers();
		this->network = network;
		this->first = network->input();
		this->last = network->output();
		this->inputs = inputs;
		this->outputs = outputs;
	}

	virtual void clean() = 0;

	virtual void fit(size_t epochs) = 0;
};

template<typename T, typename T_SIZE>
class NeuralNetwork {
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	std::vector<T_LAYER*> _layers;

public:
	NeuralNetwork(T_SIZE num_layers, ...) {
		va_list layers;
		va_start(layers, num_layers);
		for (T_SIZE i = 0; i < num_layers; i++) {
			_layers.push_back(va_arg( layers, T_LAYER* ));
		}
		va_end(layers);
	}

	inline std::vector<T_LAYER*>& layers() { return _layers; }
	inline T_LAYER* input() { return _layers.front(); }
	inline T_LAYER* output() { return _layers.back(); }

	inline void calculate() {
		for (auto & L : _layers)
			L->forward();
	}

	///
	/// Will use supplied trainer to fit NN to supplied input/output training data
	///
	/// inputs is a tensor containing training input data
	/// outputs is a tensor containing training output data (a.k.a. target)
	/// the sizes of inputs and outputs must match those of the network inputs and outputs.
	/// inputs/outputs are stacked on a depth dimention, meaning that
	/// if e.g. input depth is 3, then the depth of inputs tensor should be multiple of 3
	///  and input samples will be stacked on a depth dimention like that:
	///   [z=00, z=01, z=02, z=10, z=11, z=12, etc.], where z=ij and i is the input number and j is the z of the input tensor;
	///   basically inputs will receive a tenzor i of depth=3, where j is the z of that tensor.
	/// outputs are organized in the same manner.
	/// Additionally number of stacked input tensors must be equal to the number of stacked output tensors.
	/// in thi sexample if output layer has depth=1, then outputs should look like this:
	/// [o=10, o=20, etc.]
	void train(const T_INPUT &inputs, const T_INPUT &outputs, TrainerBase<T, T_SIZE> &trainer, size_t epochs, bool clean=true) {
		// check the sizes of inputs and outputs
		assert(inputs.depth() % input()->inputs().depth() == 0);
		assert(outputs.depth() % output()->outputs().depth() == 0);
		assert(inputs.depth() / input()->inputs().depth() ==
					 outputs.depth() / output()->outputs().depth());

		trainer.init(inputs, outputs, this);
		trainer.fit(epochs);
		if (clean)
			trainer.clean();
	}
};

};

#include <trainers/BackPropTrainer.h>

#endif
