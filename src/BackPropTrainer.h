#if !defined(ENN_BACKPROP_TRAINER_H)
#define ENN_BACKPROP_TRAINER_H

#include <NeuralNetwork.h>
#include <math.h>
#include <stdlib.h>

namespace EasyNeuralNetworks {

template<typename T, bool BIAS = ENN_DEFAULT_BIAS, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class BackPropTrainer : public TrainerBase<T, BIAS, T_SIZE> {
public:
	typedef std::function<bool (T error, size_t epoch, void * data)> EpochCallback_t;

private:
	typedef LayerBase<T, BIAS, T_SIZE>* T_LAYER;

	T * delta_out;
	T mean_error;
	T momentum;
	T decay;
	T current_momentum;
	EpochCallback_t callback;
	void * callback_data;

public:
	BackPropTrainer(T momentum, T decay, EpochCallback_t callback = NULL, void * callback_data = NULL) : TrainerBase<T, BIAS, T_SIZE>() {
		this->momentum = momentum;
		this->decay = decay;
		this->callback = callback;
		this->callback_data = callback_data;
	}

	virtual void init(const std::vector<const T*> &inputs, const std::vector<const T*> &outputs, NeuralNetwork<T, BIAS, T_SIZE>* network) override {
		TrainerBase<T, BIAS, T_SIZE>::init(inputs, outputs, network);

		delta_out = (T*)malloc(this->num_outputs * sizeof(T));

		typename std::vector<T_LAYER>::iterator I = this->layers.begin();

		while (I != this->layers.end()) {
			// initialize weights to random
			for (T_SIZE i = 0; i < (*I)->num_weights(); i++)
				(*I)->weights()[i] = .5 - (rand() / (float)RAND_MAX);
			// initialize error arrays
			if ((*I)->num_errors())
				(*I)->errors((T*)malloc(sizeof(T) * (*I)->num_errors()));
			++I;
		}

		mean_error = 0;
		current_momentum = momentum;
	}

	virtual void clean() {
		typename std::vector<T_LAYER>::iterator L = this->layers.begin();
		free(delta_out);
		// free layer error arrays
		while (L != this->layers.end()) {
			if ((*L)->errors() != NULL)
				free((*L)->errors());
			(*L)->errors(NULL);
			++L;
		}
	}

	virtual void fit(size_t epochs) {
		for (size_t epoch = 0; epoch < epochs; epoch++) {
			fit_epoch();
			// adjust parameters
			current_momentum -= current_momentum * decay;

			if (callback != NULL) {
				if (!callback(mean_error, epoch, callback_data))
					break;
			}
		}
	}

	void fit_epoch() {
		typename std::vector<const T*>::iterator I = this->inputs.begin();
		typename std::vector<const T*>::iterator O = this->outputs.begin();
		typename std::vector<T_LAYER>::reverse_iterator L;
		T_LAYER prev = NULL;
		T* errors;

		mean_error = 0;
		while (I != this->inputs.end() && O != this->outputs.end()) {
			T discrepancy ;
			// evaluate input/output pair
			memcpy(this->first->inputs(), *I, sizeof(T) * this->num_inputs);
			this->network->calculate();

			// calculate output error
			this->diff_arr(delta_out, *O, this->last->outputs(), this->num_outputs);

			// calculate error
			discrepancy = this->sqrsum_arr(delta_out, this->num_outputs);
			mean_error += discrepancy;

			++O;
			++I;

			// backpropagate
			prev = NULL;
			L = this->layers.rbegin();
			while (L != this->layers.rend()) {
				if (prev == NULL) {
					errors = delta_out;
				} else {
					errors = prev->errors();
				}
				(*L)->backward(errors);
				prev = *L;
				++L;
			}

			// update weights
			prev = NULL;
			L = this->layers.rbegin();
			while (L != this->layers.rend()) {
				if (prev == NULL) {
					errors = delta_out;
				} else {
					errors = prev->errors();
				}
				(*L)->update(errors, current_momentum);
				prev = *L;
				++L;
			}
		}
		mean_error /= (T)this->inputs.size();
	}
};

};

#endif
