#if !defined(ENN_DROPOUT_LAYER_H)
#define ENN_DROPOUT_LAYER_H

#include <LayerBase.h>
#include <LUActivation.h>
#include <stdlib.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class DropOutLayer : public LayerBase<T, BIAS, T_SIZE> {
	T * _errors;
	T _dropout_percent;
public:
	DropOutLayer(LayerBase<T, BIAS, T_SIZE>& input, T dropout_percent) : LayerBase<T, BIAS, T_SIZE>(input, input.outputs(), input.num_outputs(), LUActivation<T>()) {
		_errors = NULL;
		_dropout_percent = dropout_percent;
	}

	///
	/// returns a pointer to errors
	///
	virtual inline const T* errors() const { return _errors; }
	virtual inline T* errors() { return _errors; }
	virtual inline void errors(T * errors)  { _errors = errors; }
	virtual inline T_SIZE num_errors() const { return this->num_inputs(); }

	///
	/// returns a pointer to weights
	///
	virtual inline const T* weights() const { return NULL; };
	virtual inline T* weights() { return NULL; };
	virtual inline void weights(T * weights) { };
	virtual inline T_SIZE num_weights() const { return 0; }

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward()
	{
		T_SIZE num_drop = this->num_outputs() * _dropout_percent;
		while (num_drop) {
			T_SIZE i = this->num_outputs() * (rand() / (T)RAND_MAX);
			this->outputs()[i] = 0;
			--num_drop;
		}
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	virtual void backwards(T * deltas)
	{
		memcpy(_errors, deltas, sizeof(T) * num_errors());
	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update(const T * deltas, T alpha)
	{
	}
};

};

#endif
