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
#include <DropOutLayer.h>
#include <ConvLayer1D.h>
#include <ConvLayer2D.h>
#include <ConvLayer3D.h>
#include <PoolingLayer1D.h>
#include <ProgmemHelper.h>
#include <FixedPointType.h>

namespace EasyNeuralNetworks {

template<typename T>
class LossFunctionBase {
public:
	virtual T operator () (T * deltas, const T * target, const T * output, size_t num) const = 0;

	#define ENN_LOSS_LOOP(DELTA, ACC) \
		T acc = 0;	\
		while (num--) {	\
			*deltas = DELTA;	\
			acc += ACC;	\
			++target;	\
			++output;	\
			++deltas;	\
		}	\
		return acc;
};

template<typename T>
class L2Loss : public LossFunctionBase<T> {
public:
	virtual T operator () (T * deltas, const T * target, const T * output, size_t num) const {
		ENN_LOSS_LOOP(*output - *target, *deltas * *deltas)
	}
};

template<typename T>
class L1Loss : public LossFunctionBase<T> {
public:
	virtual T operator () (T * deltas, const T * target, const T * output, size_t num) const {
		ENN_LOSS_LOOP((*target < *output ? 1 : *target > *output ? -1 : 0), abs(*deltas))
	}
};

template<typename T>
class CrossEntropy : public LossFunctionBase<T> {
public:
	virtual T operator () (T * deltas, const T * target, const T * output, size_t num) const {
		ENN_LOSS_LOOP(*target * log(*output), -*deltas)
	}
};


template<typename T, bool BIAS = ENN_DEFAULT_BIAS, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class NeuralNetwork;

template<typename T, bool BIAS, typename T_SIZE>
class TrainerBase {
protected:
	typedef LayerBase<T, BIAS, T_SIZE>* T_LAYER;
	std::vector<const T*> inputs;
	std::vector<const T*> outputs;
	std::vector<T_LAYER> layers;
	NeuralNetwork<T, BIAS, T_SIZE>* network;

	T_SIZE num_inputs;
	T_SIZE num_outputs;
	T_LAYER first;
	T_LAYER last;

public:
	TrainerBase()
	{	}

	virtual void init(const std::vector<const T*> &inputs, const std::vector<const T*> &outputs, NeuralNetwork<T, BIAS, T_SIZE>* network) {
		this->inputs = inputs;
		this->outputs = outputs;
		this->layers = network->layers();
		this->network = network;

		this->first = network->input();
		this->last = network->output();

		this->num_inputs = first->num_inputs();
		this->num_outputs = last->num_outputs();
	}

	virtual void clean() = 0;

	virtual void fit(size_t epochs) = 0;
};

template<typename T, bool BIAS, typename T_SIZE>
class NeuralNetwork {
	typedef LayerBase<T, BIAS, T_SIZE>* T_LAYER;
	std::vector<T_LAYER> _layers;

public:
	NeuralNetwork(T_SIZE num_layers, ...) {
		va_list layers;
		va_start(layers, num_layers);
		for (T_SIZE i = 0; i < num_layers; i++) {
			_layers.push_back(va_arg( layers, T_LAYER ));
		}
		va_end(layers);

#if defined(DEBUG)
		// todo sanity checks for layer in/out
#endif
	}

	std::vector<T_LAYER>& layers() { return _layers; }
	T_LAYER input() { return _layers.front(); }
	T_LAYER output() { return _layers.back(); }

	void calculate() {
		typename std::vector<T_LAYER>::iterator p = _layers.begin();

		while (p != _layers.end()) {
			(*p)->forward();
			++p;
		}
	}

	///
	/// Will use supplied trainer to fit NN to supplied input/output training data
	///
	/// Note that the size of inputs_arr and outputs_arr must be equal to N*num_data and M*num_data
	/// where N is the number of input and M is the number of output neurons
	void train(const T * inputs_arr, const T * outputs_arr, size_t num_data, TrainerBase<T, BIAS, T_SIZE> &trainer, size_t epochs, bool clean=true) {
		std::vector<const T*> inputs;
		std::vector<const T*> outputs;

		for (size_t i = 0; i < num_data; i++) {
			inputs.push_back(inputs_arr);
			outputs.push_back(outputs_arr);
			inputs_arr += _layers.front()->num_inputs();
			outputs_arr += _layers.back()->num_outputs();
		}

		train(inputs, outputs, trainer, epochs, clean);
	}

	void train(const std::vector<const T*> &inputs, const std::vector<const T*> &outputs, TrainerBase<T, BIAS, T_SIZE> &trainer, size_t epochs, bool clean=true) {
		trainer.init(inputs, outputs, this);
		trainer.fit(epochs);
		if (clean)
			trainer.clean();
	}
};

};

#endif
