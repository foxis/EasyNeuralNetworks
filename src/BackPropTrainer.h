#if !defined(ENN_BACKPROP_TRAINER_H)
#define ENN_BACKPROP_TRAINER_H

#include <NeuralNetwork.h>
#include <math.h>
#include <stdlib.h>
#include <cstdlib.h>

namespace EasyNeuralNetworks {

template<typename T, bool BIAS = ENN_DEFAULT_BIAS, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class BackPropTrainer : public TrainerBase<T, BIAS, T_SIZE> {
	typedef LayerBase<T, BIAS, T_SIZE>* T_LAYER;

	std::vector<T *> delta_out;
	T mean_error;

public:
	BackPropTrainer() : TrainerBase<T, BIAS, T_SIZE>() {

	}

	virtual void init(const std::vector<const T*> &inputs, const std::vector<const T*> &outputs, std::vector<T_LAYER> &layers) {
		TrainerBase<T, BIAS, T_SIZE>::init(inputs, outputs, layers);

		for (size_t i = 0; i < inputs.size(); i++)
			delta_out.push_back((T*)malloc(num_outputs * sizeof(T)));

		typename std::vector<T_LAYER>::iterator I = layers.begin();

		// initialize weights to random
		while (I != layers.end()) {
			for (T_SIZE i = 0; i < (*I)->num_weights(); i++) {
				(*I)->weights()[i] = .5 - (rand() / (T)RAND_MAX);
			}
			++I;
		}

		// initialize error arrays

		mean_error = 0;
	}

	virtual void clean() {
		typename std::vector<T*>::iterator I = delta_out.begin();
		while (I != delta_out.end()) {
			free(*I);
			++I;
		}
		// free layer error arrays
	}

	virtual void fit(size_t epochs) {
		for (size_t epoch = 0; epoch < epochs; epoch++) {
			fit_epoch();
			// adjust parameters
			// call callback
			// check termination criteria
		}
	}

	void fit_epoch() {
		typename std::vector<const T*>::iterator I = inputs.begin();
		typename std::vector<const T*>::iterator O = outputs.begin();
		typename std::vector<T*>::iterator delta = delta_out.begin();

		mean_error = 0;
		while (I != inputs.end() && O != outputs.eng()) {
			T discrepancy ;
			// evaluate input/output pair
			memcpy(first.inputs(), *I, sizeof(T) * num_inputs);
			evaluate();

			// calculate output error
			diff_arr(*delta, *O, last.outputs());

			// calculate error
			discrepancy = sqrsum_arr(*delta);
			mean_error += discrepancy;

			++O;
			++I;
			++delta;
		}
		mean_error /= (T)inputs.size();

		// backpropagate
		delta = delta_out.begin();
		while (delta != delta_out.end()) {
			typename std::vector<T_LAYER>::reverse_iterator L = layers.rbegin();
			T_LAYER prev = NULL;
			T* errors;

			while (L != layers.rend()) {
				if (prev == NULL) {
					errors = *delta;
				} else {
					errors = prev.errors();
				}
				(*L)->backwards(errors);
				prev = *L;
				++L;
			}
			++delta;
		}

		// update weights
		typename std::vector<T_LAYER>::iterator L = layers.begin();
		while (L != layers.end()) {
			(*L)->update();
		}
	}
};

};

#endif
