#if !defined(ENN_LAYER_BASE_H)
#define ENN_LAYER_BASE_H

#if !defined(ENN_DEFAULT_TYPE)
#define ENN_DEFAULT_TYPE float
#endif

#if !defined(ENN_DEFAULT_SIZE_TYPE)
#define ENN_DEFAULT_SIZE_TYPE uint16_t
#endif

#if !defined(ENN_DEFAULT_ACTIVATION)
#define ENN_DEFAULT_ACTIVATION(TYPE) ReLUActivation<TYPE>
#endif

#if !defined(ENN_DEFAULT_BIAS)
#define ENN_DEFAULT_BIAS true
#endif

namespace EasyNeuralNetworks {

template<typename T>
class ActivationBase {
public:
	virtual T forward(T val) const = 0;
	virtual T backward(T val) const = 0;

	inline void apply_forward_inplace(T * a, size_t num) const {
			for (size_t i = 0; i < num; i++) {
				*a = forward(*a);
				++a;
			}
	}

	inline void apply_backward_inplace(T * a, size_t num) const {
		for (size_t i = 0; i < num; i++) {
			*a = backward(*a);
			++a;
		}
	}
};

///
/// A abstract base layer class
/// stores pointers to layer inputs and outputs along with their sizes
///
template <typename T, bool BIAS, typename T_SIZE>
class LayerBase {
protected:
	T_SIZE _num_inputs;
	T_SIZE _num_outputs;
	T* _inputs;
	T* _outputs;
	const ActivationBase<T> &_activation;

public:
	///
	///
	///
	LayerBase(T* inputs, T_SIZE num_inputs, T* outputs, T_SIZE num_outputs, const ActivationBase<T>& activation)
	 		: _activation(activation) {
		_num_inputs = num_inputs;
		_num_outputs = num_outputs;
		_inputs = inputs;
		_outputs = outputs;
	}

	///
	///
	///
	LayerBase(LayerBase &input, T* outputs, T_SIZE num_outputs, const ActivationBase<T>& activation)
		: _activation(activation) {
		_num_inputs = input.num_outputs();
		_inputs = input.outputs();
		_num_outputs = num_outputs;
		_outputs = outputs;
	}

	///
	///
	///
	inline const ActivationBase<T>& activation() const { return _activation; }
	inline void activation(const ActivationBase<T>& act) const { _activation = act; }

	///
	/// returns a const pointer to inputs
	///
	inline const T* inputs() const { return _inputs; }
	inline T* inputs() { return _inputs; }
	inline T_SIZE num_inputs() const { return _num_inputs; }
	inline void inputs(T * inputs) { _inputs = inputs; }

	///
	/// returns a pointer to outputs
	///
	inline T* outputs() { return _outputs; }
	inline const T* outputs() const { return _outputs; }
	inline T_SIZE num_outputs() const { return _num_outputs; }
	inline void outputs(T * outputs) { _outputs = outputs; }

	///
	/// returns a pointer to errors
	///
	virtual const T* errors() const = 0;
	virtual T* errors() = 0;
	virtual void errors(T * errors) = 0;
	virtual T_SIZE num_errors() const = 0;

	///
	/// returns a pointer to weights
	///
	virtual const T* weights() const = 0;
	virtual T* weights() = 0;
	virtual void weights(T * weights) = 0;
	virtual T_SIZE num_weights() const = 0;

	///
	/// performs a forward calculation
	/// outputs() will write the result in output data
	virtual void forward() = 0;

	///
	/// performs error back propagation.
	/// will calculate errors for the inputs.
	/// will modify errors inplace
	virtual void backwards(T * deltas) = 0;

	///
	/// will update the weights calculated in backwards
	///
	/// alpha is the multiplier determining how much to adjust the weights
	/// based on errors
	virtual void update(const T * deltas, T alpha) = 0;


	///
	/// will calculate the output for 1d convolutional network
	/// NOTE:
	///   Ii, where i < N
	///   Wij where i < width and j < K
	///   Ojk where j < K, k < N - width
	///   N - number of inputs
	///	  K - number of kernels
	///   width - kernel size
	///
	///   Okj = SUMi I(i + k) * Wij
	///
	///  Wij = W_arr(i + j * N)
	///  Okj = O_arr(k + j * (N - width))
	inline void calc_conv_1d(const T * wights, T_SIZE width, T_SIZE kernels) {
		T acc;
		T_SIZE i, j, k;
		const T * in_p;
		const T * w_p;
		T * out_p;
		T_SIZE NW = _num_inputs - width;

		// iterate over kernels
		for (j = 0, out_p = _outputs; j < kernels; j++) {
			// convolve
			for (k = 0; k < NW; k++) {
				acc = 0;
				w_p = weights + j * width;
				in_p = _inputs + k;

				for (i = 0; i < width; i++) {
					acc += (*in_p) * (*w_p);

					++in_p;
					++w_p;
				}

				if (BIAS) {
					acc += (*w_p);
					++w_p;
				}
				*out_p = _activation.forward(acc);
				++out_p;
			}
		}
	}

	///
	/// will calculate the output for 2d convolutional network
	/// TODO
	inline void calc_conv_2d(const T * kernel, T_SIZE width, T_SIZE height, bool bias) {
		T acc;
		T_SIZE out_i;
		T_SIZE in_i;
		const T * in_p;
		const T * w_p;
		T * out_p;
	}
};

};

#endif
