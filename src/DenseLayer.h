#if !defined(ENN_DENSE_LAYER_H)
#define ENN_DENSE_LAYER_H

#include <LayerBase.h>
#include <ReLUActivation.h>
#include <SigmoidActivation.h>
#include <ProgmemHelper.h>

namespace EasyNeuralNetworks {

template <typename T = ENN_DEFAULT_TYPE,
				  bool BIAS = ENN_DEFAULT_BIAS,
					typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class DenseLayer : public LayerBase<T, BIAS, T_SIZE> {
	T * _weights;
public:
	DenseLayer(T_SIZE num_inputs, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(NULL, num_inputs, NULL, num_outputs, activation) {
		_weights = (T*)malloc(sizeof(T) * (num_inputs + (BIAS?1:0)) * num_outputs);
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
		this->inputs((T*)malloc(sizeof(T) * num_inputs));
	}
	DenseLayer(const LayerBase<T, BIAS, T_SIZE> &input, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		_weights = (T*)malloc(sizeof(T) * (input.num_inputs() + (BIAS?1:0)) * num_outputs);
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}
	DenseLayer(const LayerBase<T, BIAS, T_SIZE> &input, T * weights, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		_weights = weights;
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}
	DenseLayer(const LayerBase<T, BIAS, T_SIZE> &input, const ProgmemHelper<T> & weights, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: LayerBase<T, BIAS, T_SIZE>(input, NULL, num_outputs, activation) {
		_weights = weights.read((input.num_inputs() + (BIAS?1:0)) * num_outputs);
		this->outputs((T*)malloc(sizeof(T) * num_outputs));
	}

	///
	/// returns a pointer to errors
	///
	virtual inline T* errors() { return NULL; }

	///
	/// returns a pointer to weights
	///
	virtual inline const T* weights() const { return _weights; };
	virtual inline T* weights() { return _weights; };
	virtual inline void weights(T * weights) { _weights = weights; };

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward()
	{
		this->calc_dense(_weights);
	}

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	///
	virtual void backwards(const T * errors)
	{

	}

	///
	/// will update the weights calculated in backwards
	///
	virtual void update()
	{

	}

};

};

#endif
