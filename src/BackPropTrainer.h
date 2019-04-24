#if !defined(ENN_BACKPROP_TRAINER_H)
#define ENN_BACKPROP_TRAINER_H

#include <NeuralNetwork.h>
#include <math.h>
#include <stdlib.h>

namespace EasyNeuralNetworks {

template<typename T, bool BIAS = ENN_DEFAULT_BIAS, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class BackPropTrainer : public TrainerBase<T, BIAS, T_SIZE> {
	typedef LayerBase<T, BIAS, T_SIZE>* T_LAYER;

	std::vector<T *> delta_out;
	T mean_error;
	T alpha;
	T beta;
	T current_alpha;

public:
	BackPropTrainer(T alpha, T beta) : TrainerBase<T, BIAS, T_SIZE>() {
		this->alpha = alpha;
		this->beta = beta;
	}

	virtual void init(const std::vector<const T*> &inputs, const std::vector<const T*> &outputs, std::vector<T_LAYER> &layers) {
		TrainerBase<T, BIAS, T_SIZE>::init(inputs, outputs, layers);

		for (size_t i = 0; i < inputs.size(); i++)
			delta_out.push_back((T*)malloc(this->num_outputs * sizeof(T)));

		typename std::vector<T_LAYER>::iterator I = this->layers.begin();

		while (I != layers.end()) {
			// initialize weights to random
			for (T_SIZE i = 0; i < (*I)->num_weights(); i++) {
				(*I)->weights()[i] = .5 - (rand() / (T)RAND_MAX);
			}
			// initialize error arrays
			if ((*I)->num_errors())
				(*I)->errors((T*)malloc(sizeof(T) * (*I)->num_errors()));
			++I;
		}

		mean_error = 0;
		current_alpha = alpha;
	}

	virtual void clean() {
		typename std::vector<T*>::iterator I = delta_out.begin();
		typename std::vector<T_LAYER>::iterator L = this->layers.begin();
		while (I != delta_out.end()) {
			free(*I);
			++I;
		}
		// free layer error arrays
		while (L != this->layers.end()) {
			if ((*I)->errors() != NULL)
				free((*L)->errors());
			(*L)->errors(NULL);
			++L;
		}
	}

	virtual void fit(size_t epochs) {
		for (size_t epoch = 0; epoch < epochs; epoch++) {
			fit_epoch();
			// adjust parameters
			current_alpha -= current_alpha * beta;
			// call callback
			// check termination criteria
		}
	}

	void fit_epoch() {
		typename std::vector<const T*>::iterator I = this->inputs.begin();
		typename std::vector<const T*>::iterator O = this->outputs.begin();
		typename std::vector<T*>::iterator delta = delta_out.begin();

		mean_error = 0;
		while (I != this->inputs.end() && O != this->outputs.eng()) {
			T discrepancy ;
			// evaluate input/output pair
			memcpy(this->first->inputs(), *I, sizeof(T) * this->num_inputs);
			this->evaluate();

			// calculate output error
			diff_arr(*delta, *O, this->last->outputs());

			// calculate error
			discrepancy = sqrsum_arr(*delta);
			mean_error += discrepancy;

			++O;
			++I;
			++delta;
		}
		mean_error /= (T)this->inputs.size();

		// backpropagate
		delta = delta_out.begin();
		while (delta != delta_out.end()) {
			typename std::vector<T_LAYER>::reverse_iterator L = this->layers.rbegin();
			T_LAYER prev = NULL;
			T* errors;

			while (L != this->layers.rend()) {
				if (prev == NULL) {
					errors = *delta;
				} else {
					errors = prev.errors();
				}
				(*L)->backwards(errors);
				prev = *L;
				++L;
			}

			// update weights
			prev = NULL;
			L = this->layers.rbegin();
			while (L != this->layers.rend()) {
				if (prev == NULL) {
					errors = *delta;
				} else {
					errors = prev.errors();
				}
				(*L)->update(errors, current_alpha);
				prev = *L;
				++L;
			}

			++delta;
		}
	}
};

};

#endif
