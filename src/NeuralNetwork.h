#if !defined(ENN_NEURAL_NETWORK_H)
#define ENN_NEURAL_NETWORK_H

#include <LayerBase.h>
#include <cstdarg>
#include <vector>

#include <LUActivation.h>
#include <ReLUActivation.h>
#include <SigmoidActivation.h>
#include <TanhActivation.h>
#include <InputLayer.h>
#include <DenseLayer.h>
#include <ProgmemHelper.h>

namespace EasyNeuralNetworks {

template<typename T, BIAS, T_SIZE>
class TrainerBase {
	typedef LayerBase<T, BIAS, T_SIZE>* T_LAYER;
	std::vector<T_LAYER> _layers;

public:
	void init(const std::vector<const T*> &inputs, const std::vector<const T*> &outputs, std::vector<T_LAYER> layers) = 0;

	void fit(size_t epochs);
};

template<typename T, bool BIAS = ENN_DEFAULT_BIAS, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
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
	void train(const T * inputs_arr, const T * outputs_arr, size_t num_data, TrainerBase<T> &trainer, size_t epochs) {
		std::vector<const T*> inputs;
		std::vector<const T*> outputs;

		for (size_t i = 0; i < num_data) {
			inputs.push_back(inputs_arr);
			outputs.push_back(outputs_arr);
			inputs_arr + _layers.front()->num_inputs();
			outputs_arr + _layers.back()->num_outputs();
		}

		train(inputs, outputs, trainer, epochs);
	}

	void train(const std::vector<const T*> &inputs, const std::vector<const T*> &outputs, size_t epochs) {
		trainer.init(inputs, outputs, _layers);
		trainer.fit(epochs);
	}
};

};

#endif
