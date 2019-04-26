#if !defined(ENN_BACKPROP_TRAINER_H)
#define ENN_BACKPROP_TRAINER_H

#include <NeuralNetwork.h>
#include <matvecop.h>

namespace EasyNeuralNetworks {

enum ENN_WEIGHT_INITIALIZERS {
	ENN_WEIGHTS_NONE = 0,
	ENN_WEIGHTS_FLAT,
	ENN_WEIGHTS_NORMAL,
	ENN_WEIGHTS_XAVIER,
};

template<typename T, ENN_WEIGHT_INITIALIZERS W_INIT = ENN_WEIGHTS_FLAT, bool BIAS = ENN_DEFAULT_BIAS, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
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
	const LossFunctionBase<T>& loss_func;
	EpochCallback_t callback;
	void * callback_data;

public:
	BackPropTrainer() : BackPropTrainer(2, .001, L2Loss<T>(), NULL, NULL) {}
	BackPropTrainer(EpochCallback_t callback, void * callback_data=NULL) : BackPropTrainer(2, .001, L2Loss<T>(), callback, callback_data) {}
	BackPropTrainer(T momentum, T decay) : BackPropTrainer(momentum, decay, L2Loss<T>(), NULL, NULL) {}
	BackPropTrainer(T momentum, T decay, EpochCallback_t callback, void * callback_data=NULL) : BackPropTrainer(momentum, decay, L2Loss<T>(), callback, callback_data) {}
	BackPropTrainer(T momentum, T decay, const LossFunctionBase<T>& loss_func) : BackPropTrainer(momentum, decay, loss_func, NULL, NULL) {}
	BackPropTrainer(T momentum, T decay, const LossFunctionBase<T>& loss_func, EpochCallback_t callback, void * callback_data = NULL)
		: TrainerBase<T, BIAS, T_SIZE>(),
		 loss_func(loss_func) {
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
			if (W_INIT != ENN_WEIGHTS_NONE) {
				// initialize weights
				for (T_SIZE i = 0; i < (*I)->num_weights(); i++)
					switch (W_INIT) {
						case ENN_WEIGHTS_FLAT:
							(*I)->weights()[i] = random_flat(0, 1.0);
							break;
						case ENN_WEIGHTS_NORMAL:
							(*I)->weights()[i] = random_normal(0, 1);
							break;
						case ENN_WEIGHTS_XAVIER:
							(*I)->weights()[i] = random_normal(0, 1) * sqrt(2 / (*I)->num_weights());
							break;
					}
			}

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
			// evaluate input/output pair
			memcpy(this->first->inputs(), *I, sizeof(T) * this->num_inputs);
			this->network->calculate();

			// calculate output error
			mean_error += loss_func(delta_out, *O, this->last->outputs(), this->num_outputs);

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
