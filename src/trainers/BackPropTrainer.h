#if !defined(ENN_BACKPROP_TRAINER_H)
#define ENN_BACKPROP_TRAINER_H

#include <NeuralNetwork.h>
#include <core/matvecop.h>

namespace EasyNeuralNetworks {

enum ENN_WEIGHT_INITIALIZERS {
	ENN_WEIGHTS_NONE = 0,
	ENN_WEIGHTS_FLAT,
	ENN_WEIGHTS_NORMAL,
	ENN_WEIGHTS_XAVIER,
};

template<typename T, ENN_WEIGHT_INITIALIZERS W_INIT = ENN_WEIGHTS_FLAT, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class BackPropTrainer : public TrainerBase<T, T_SIZE> {
public:
	typedef std::function<bool (T error, size_t epoch, void * data)> EpochCallback_t;

private:
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	ENN_T_LOSS_TYPEDEF(T_LOSS);

	T_INPUT *out_gradients;
	T mean_error;
	T momentum;
	T decay;
	T current_momentum;
	const T_LOSS& loss_func;
	EpochCallback_t callback;
	void * callback_data;

public:
	BackPropTrainer() : BackPropTrainer(2, .001, L2Loss<T, T_SIZE>(), NULL, NULL) {}
	BackPropTrainer(EpochCallback_t callback, void * callback_data=NULL) : BackPropTrainer(2, .001, L2Loss<T>(), callback, callback_data) {}
	BackPropTrainer(T momentum, T decay) : BackPropTrainer(momentum, decay, L2Loss<T, T_SIZE>(), NULL, NULL) {}
	BackPropTrainer(T momentum, T decay, EpochCallback_t callback, void * callback_data=NULL) : BackPropTrainer(momentum, decay, L2Loss<T, T_SIZE>(), callback, callback_data) {}
	BackPropTrainer(T momentum, T decay, const T_LOSS& loss_func) : BackPropTrainer(momentum, decay, loss_func, NULL, NULL) {}
	BackPropTrainer(T momentum, T decay, const T_LOSS& loss_func, EpochCallback_t callback, void * callback_data = NULL)
		: TrainerBase<T, T_SIZE>(),
		 loss_func(loss_func) {
		this->momentum = momentum;
		this->decay = decay;
		this->callback = callback;
		this->callback_data = callback_data;
	}

	virtual void init(const T_INPUT &inputs, const T_INPUT &outputs, NeuralNetwork<T, T_SIZE>* network) override {
		TrainerBase<T, T_SIZE>::init(inputs, outputs, network);

		out_gradients = this->last->outputs().clone_new(false);

		for (auto L : this->layers) {
			L->training_begin();

			if (W_INIT != ENN_WEIGHTS_NONE && L->trainable()) {
				// initialize weights
				auto I = L->weights().begin(1);
				auto num = L->weights().size();
				while (num--) {
					switch (W_INIT) {
						case ENN_WEIGHTS_FLAT:
							*I = random_flat(0, 1.0);
							break;
						case ENN_WEIGHTS_NORMAL:
							*I = random_normal(0, 1);
							break;
						case ENN_WEIGHTS_XAVIER:
							*I = random_normal(0, 1) * sqrt(2 / (T)num);
							break;
					}
					++I;
				}
			}
		}

		mean_error = 0;
		current_momentum = momentum;
	}

	virtual void clean() {
		delete out_gradients;

		for (auto L : this->layers)
			L->training_end();
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
		typename std::vector<T_LAYER*>::reverse_iterator L;
		T_INPUT *gradients;

		mean_error = 0;

		for (T_SIZE input_output_pair = 0; input_output_pair < this->inputs.depth(); input_output_pair++) {
			// evaluate input/output pair
			this->first->inputs().copy(this->inputs.window(input_output_pair * this->first->inputs().depth(), this->first->inputs().depth()));

			this->network->calculate();

			// calculate output error
			mean_error += loss_func(*out_gradients, this->outputs.window(input_output_pair * this->last->outputs().depth(), this->last->outputs().depth()), this->last->outputs());

			// backpropagate
			L = this->layers.rbegin();
			gradients = out_gradients;
			while (L != this->layers.rend()) {
				(*L)->backward(*gradients);
				gradients = &(*L)->gradients();
				++L;
			}

			// update weights
			L = this->layers.rbegin();
			gradients = out_gradients;
			while (L != this->layers.rend()) {
				(*L)->update(*gradients, current_momentum);
				gradients = &(*L)->gradients();
				++L;
			}
		}
		mean_error /= (T)this->inputs.size();
	}
};

};

#endif
