#if !defined(ENN_NEURAL_NETWORK_H)
#define ENN_NEURAL_NETWORK_H

#include <LayerBase.h>
#include <cstdarg>
#include <vector>
#include <stdlib.h>

#include <LUActivation.h>
#include <ReLUActivation.h>
#include <SigmoidActivation.h>
#include <TanhActivation.h>
#include <SoftplusActivation.h>
#include <InputLayer.h>
#include <DenseLayer.h>
//#include <DropOutLayer.h>
//#include <ConvLayer1D.h>
//#include <ConvLayer2D.h>
//#include <ConvLayer3D.h>
//#include <PoolingLayer1D.h>
#include <FixedPointType.h>

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
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	tensor<float, T_SIZE> inputs;
	tensor<float, T_SIZE> outputs;
	std::vector<T_LAYER*> layers;
	NeuralNetwork<T, T_SIZE>* network;

	T_SIZE num_inputs;
	T_SIZE num_outputs;
	T_LAYER *first;
	T_LAYER *last;

public:
	TrainerBase()
	{	}

	virtual void init(const tensor<float, T_SIZE> &inputs, const tensor<float, T_SIZE> &outputs, NeuralNetwork<T, T_SIZE>* network) {
		this->inputs = inputs;
		this->outputs = outputs;
		this->layers = network->layers();
		this->network = network;

		this->first = network->input();
		this->last = network->output();

		this->num_inputs = first->inputs().size();
		this->num_outputs = last->outputs().size();
	}

	virtual void clean() = 0;

	virtual void fit(size_t epochs) = 0;
};

template<typename T, typename T_SIZE>
class NeuralNetwork {
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	std::vector<T_LAYER*> _layers;

public:
	NeuralNetwork(T_SIZE num_layers, ...) {
		va_list layers;
		va_start(layers, num_layers);
		for (T_SIZE i = 0; i < num_layers; i++) {
			_layers.push_back(va_arg( layers, T_LAYER* ));
		}
		va_end(layers);

#if defined(DEBUG)
		// todo sanity checks for layer in/out
#endif
	}

	std::vector<T_LAYER*>& layers() { return _layers; }
	T_LAYER* input() { return _layers.front(); }
	T_LAYER* output() { return _layers.back(); }

	void calculate() {
		for (auto * L : _layers)
			L->forward();
	}

	///
	/// Will use supplied trainer to fit NN to supplied input/output training data
	///
	/// Note that the size of inputs_arr and outputs_arr must be equal to N*num_data and M*num_data
	/// where N is the number of input and M is the number of output neurons
	void train(const tensor<float, T_SIZE> &inputs, const tensor<float, T_SIZE> &outputs, TrainerBase<T, T_SIZE> &trainer, size_t epochs, bool clean=true) {
		assert(inputs.depth() == outputs.depth() &&
			inputs.width() == input()->inputs().width() && inputs.height() == input()->inputs().height() &&
			outputs.width() == output()->outputs().width() && outputs.height() == output()->outputs().height());
		trainer.init(inputs, outputs, this);
		trainer.fit(epochs);
		if (clean)
			trainer.clean();
	}
};

};

#endif
