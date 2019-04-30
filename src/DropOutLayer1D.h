#if !defined(ENN_DROPOUT_LAYER_1D_H)
#define ENN_DROPOUT_LAYER_1D_H

#include <LayerBase.h>
#include <LUActivation.h>
#include <stdlib.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class DropOutLayer1D : public LayerBase<T, T_SIZE> {
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	T _dropout_percent;
public:
	DropOutLayer1D(T_LAYER& input, T dropout_percent) : DropOutLayer1D(input.inputs(), dropout_percent) {}
	DropOutLayer1D(T_INPUT& input, T dropout_percent) : T_LAYER(input, LUActivation<T>()) {
		this->outputs(this->inputs());
		_dropout_percent = dropout_percent;
	}

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward()
	{
		T_SIZE height = this->inputs().height();
		T_SIZE depth = this->inputs().depth();
		this->weights().fill(1);
		while (depth) {
			T_SIZE num_drop = height * _dropout_percent;
			while (num_drop) {
				T_SIZE i = (height - 1) * (rand() / (T)RAND_MAX);
				this->weights().fill(i, depth - 1, 0);
				--num_drop;
			}
			--depth;
		}
		hadamard_product<T, T_SIZE>(this->outputs(), this->inputs(), this->weights(), this->outputs().size());
	}

	virtual void training_begin()
	{
		this->outputs().resize(this->inputs());
		this->weights().resize(this->inputs());
		this->gradients().resize(this->inputs());
	}
	virtual void training_end()
	{
		this->gradients().resize(0, 0, 0);
		this->outputs(this->inputs());
		this->weights().resize(0, 0, 0);
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	virtual void backward(T_INPUT& gradients)
	{
		hadamard_product<T, T_SIZE>(this->gradients(), gradients, this->weights(), this->inputs().size());
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T_INPUT& gradients, T alpha) { }
};

};

#endif
