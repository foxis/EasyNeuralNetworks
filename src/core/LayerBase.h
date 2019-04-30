#if !defined(ENN_LAYER_BASE_H)
#define ENN_LAYER_BASE_H

#if !defined(ENN_DEFAULT_ACTIVATION)
#define ENN_DEFAULT_ACTIVATION(TYPE) ReLUActivation<TYPE>
#endif

#if !defined(ENN_DEFAULT_BIAS)
#define ENN_DEFAULT_BIAS true
#endif

#include <core/tensor.h>

namespace EasyNeuralNetworks {

template<typename T, typename T_SIZE>
class ActivationBase {
public:
	virtual T forward(T val) const = 0;
	virtual T backward(T val) const = 0;

	virtual void apply_forward_inplace(tensor<T, T_SIZE>& a) const {
		auto I = a.data();
		auto num = a.size();

		while (num--) {
			*I = forward(*I);
			++I;
		}
	}

	virtual void apply_backward_inplace(tensor<T, T_SIZE>& deltas, const tensor<T, T_SIZE>& outputs) const {
		assert(deltas.size() == outputs.size());
		auto D = deltas.data();
		auto O = outputs.data();
		auto num = outputs.size();
		while (num--) {
			*D = *D * backward(*O);
			++O;
			++D;
		}
	}
};

#define ENN_T_INPUT_TYPEDEF(T_INPUT_NAME) typedef tensor<T, T_SIZE> T_INPUT_NAME;
#define ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION_NAME) typedef ActivationBase<T, T_SIZE> T_ACTIVATION_NAME;
#define ENN_T_LAYER_TYPEDEF(T_LAYER_NAME) typedef LayerBase<T, T_SIZE> T_LAYER_NAME;
///
/// A abstract base layer class
/// stores pointers to layer inputs and outputs along with their sizes
///
template <typename T, typename T_SIZE>
class LayerBase {
protected:
	ENN_T_LAYER_TYPEDEF(T_LAYER);
	ENN_T_INPUT_TYPEDEF(T_INPUT);
	ENN_T_ACTIVATION_TYPEDEF(T_ACTIVATION);
	T_INPUT _inputs;
	T_INPUT _outputs;
	T_INPUT _weights;
	T_INPUT _gradients;
	const T_ACTIVATION& _activation;
	bool _trainable = true;
public:
	LayerBase(const T_ACTIVATION& activation) : _activation(activation) {	}

	LayerBase(T_LAYER& input, const T_ACTIVATION& activation)
	 		: _inputs(input.outputs()), _activation(activation) {	}

	LayerBase(T_INPUT& inputs, const T_ACTIVATION& activation)
	 		: _inputs(inputs), _activation(activation) {	}

	LayerBase(T_LAYER& input, T_LAYER& output, T_INPUT& weights, const T_ACTIVATION& activation)
	 		: _inputs(input.outputs()), _outputs(output.inputs()), _weights(weights), _activation(activation) {	}

	LayerBase(T_INPUT& inputs, T_INPUT& outputs, T_INPUT& weights, const T_ACTIVATION& activation)
	 		: _inputs(inputs), _outputs(outputs), _weights(weights), _activation(activation) {	}

	///
	///
	///
	inline const T_ACTIVATION& activation() const { return _activation; }
	inline void activation(T_ACTIVATION& act) const { _activation = act; }

	inline bool trainable() const { return _trainable; }
	inline void trainable(bool trainable) { _trainable = trainable; }

	///
	/// gradients calculated for the inputs
	///
	virtual inline const T_INPUT& gradients() const { return _gradients; }
	virtual inline T_INPUT& gradients() { return _gradients; }
	virtual inline void gradients(T_INPUT& gradients)  { _gradients = gradients; }

	///
	///
	///
	inline const T_INPUT& inputs() const { return _inputs; }
	inline T_INPUT& inputs() { return _inputs; }
	inline void inputs(T_INPUT& inputs) { _inputs = inputs; }

	///
	///
	///
	inline T_INPUT& outputs() { return _outputs; }
	inline const T_INPUT& outputs() const { return _outputs; }
	inline void outputs(T_INPUT& outputs) { _outputs = outputs; }

	///
	///
	///
	virtual inline const T_INPUT& weights() const { return _weights; };
	virtual inline T_INPUT& weights() { return _weights; };
	virtual inline void weights(T_INPUT& weights) { _weights = weights; };

	///
	/// performs a forward calculation
	/// outputs() will write the result in output
	virtual void forward() = 0;

	virtual void training_begin() = 0;
	virtual void training_end() = 0;

	///
	/// performs error back propagation.
	/// will calculate gradients for the inputs.
	/// will modify deltas inplace
	virtual void backward(T_INPUT& deltas) = 0;

	///
	/// will update the weights based on gradients
	///
	virtual void update(const T_INPUT& gradients, T alpha) = 0;
};

};

#endif
