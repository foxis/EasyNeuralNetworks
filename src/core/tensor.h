#if !defined(ENN_TENSOR_H)
#define ENN_TENSOR_H

#include <iterator>
#include <assert.h>


#if !defined(ENN_DEFAULT_TYPE)
#define ENN_DEFAULT_TYPE float
#endif

#if !defined(ENN_DEFAULT_SIZE_TYPE)
#define ENN_DEFAULT_SIZE_TYPE uint16_t
#endif

#include <core/ProgmemHelper.h>

namespace EasyNeuralNetworks {

///
/// implements a three dimentional tensor.
///
template<typename T, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class tensor {
protected:
	T * _data;
	bool _needs_free;
	T_SIZE _width;
	T_SIZE _height;
	T_SIZE _depth;

public:
	template<bool IS_CONST>
	class iterator_ : public std::iterator<std::forward_iterator_tag, T>{
		typedef typename std::conditional<IS_CONST, const T*, T*>::type T_TYPE;
		typedef typename std::conditional<IS_CONST, const T&, T&>::type T_TYPE_REF;

		T_TYPE ptr;
		T_SIZE stride;
	public:
		iterator_() : iterator_<IS_CONST>(NULL, 0) {}
		iterator_(iterator_<IS_CONST> &i) : iterator_<IS_CONST>(i, i._stride) {}
		iterator_(iterator_<IS_CONST> &i, T_SIZE stride) : iterator_<IS_CONST>(i.ptr, stride) {}
		iterator_(T_TYPE ptr, T_SIZE stride) : ptr(ptr), stride(stride) { }
		iterator_(const iterator_<false>& other) : ptr(other.ptr), stride(other.stride) {}
		iterator_(const iterator_<true>& other) : ptr(other.ptr), stride(other.stride) {}
		~iterator_() {}

		inline iterator_<IS_CONST>  operator++(int) /* postfix */         { iterator_<IS_CONST> tmp(ptr, stride); ptr += stride; return tmp; }
    inline iterator_<IS_CONST>& operator++()    /* prefix */          { ptr += stride; return *this; }
		inline T_TYPE_REF operator* () const                    { return *ptr; }
		inline T_TYPE_REF operator* ()                    { return *ptr; }
    inline T_TYPE   operator->() const                    { return ptr; }
		inline iterator_<IS_CONST>  operator+ (T_SIZE v)   const { return ptr + v; }
		inline void operator >>= (T_SIZE v) { ptr += v; }
		inline void operator <<= (T_SIZE v) { ptr -= v; }

    inline bool      operator==(const iterator_<IS_CONST>& rhs) const { return ptr == rhs.ptr; }
    inline bool      operator!=(const iterator_<IS_CONST>& rhs) const { return ptr != rhs.ptr; }

		friend class iterator_<true>;
	};

	template<bool IS_CONST>
	class range_ {
		iterator_<IS_CONST> _begin;
		iterator_<IS_CONST> _end;
	public:
		range_(range_<IS_CONST> &r) : _begin(r._begin), _end(r._end) {}
		range_(T * ptr, T_SIZE stride, T_SIZE size)
			: _begin(ptr, stride), _end(ptr + size, stride) {
		}

		range_(const range_<false>& other) : _begin(other._begin), _end(other._end) {}
		range_(const range_<true>& other) : _begin(other._begin), _end(other._end) {}

		inline iterator_<IS_CONST> begin() const { return _begin; }
		inline iterator_<IS_CONST> end() const { return _end; }

		inline void operator += (T_SIZE v) { _begin += v; _end += v; }

		inline void operator >>= (T_SIZE v) { _begin >>= v; _end >>= v; }

		friend class range_<true>;
	};

	typedef iterator_<true> const_iterator;
	typedef iterator_<false> iterator;
	typedef range_<true> const_range;
	typedef range_<true> range;
public:
	tensor() : tensor(NULL, 0, 0, 0) {}
	tensor(T_SIZE width) : tensor(width, 1) {}
	tensor(T_SIZE width, T_SIZE height) : tensor(width, height, 1) {}
	tensor(T_SIZE width, T_SIZE height, T_SIZE depth)
		: tensor(new T[width * height * depth], width, height, depth) {
		this->_needs_free = true;
	}
	tensor(T * data, T_SIZE width) : tensor(data, width, 1) {}
	tensor(T * data, T_SIZE width, T_SIZE height) : tensor(data, width, height, 1) {}
	tensor(T * data, T_SIZE width, T_SIZE height, T_SIZE depth) {
		this->_data = data;
		this->_width = width;
		this->_height = height;
		this->_depth = depth;
		this->_needs_free = false;
	}
	tensor(const ProgmemHelper<T> & data, T_SIZE width, T_SIZE height, T_SIZE depth)
		: tensor(new T[width * height * depth], width, height, depth) {
		this->_needs_free = true;
		data.read(this->_data, width * height * depth);
	}
	tensor(const tensor<T, T_SIZE>& t) {
		this->_data = NULL;
		*this = t;
	}

	~tensor() {
		resize(0, 0, 0);
	}

	inline void operator = (const tensor<T, T_SIZE>& t) {
		if (this->_data != NULL && this->_needs_free)
			delete[] this->_data;
		this->_data = t._data;
		this->_width = t._width;
		this->_height = t._height;
		this->_depth = t._depth;
		this->_needs_free = false;
	}

	/// raw access stuff
	inline T* data() { return _data; }
	inline T* data(T_SIZE z) { return _data + offset(z); }
	inline T* data(T_SIZE y, T_SIZE z) { return _data + offset(y, z); }
	inline T* data(T_SIZE x, T_SIZE y, T_SIZE z) { return _data + offset(x, y, z); }

	inline const T* data() const { return _data; }
	inline const T* data(T_SIZE z) const { return _data + offset(z); }
	inline const T* data(T_SIZE y, T_SIZE z) const { return _data + offset(y, z); }
	inline const T* data(T_SIZE x, T_SIZE y, T_SIZE z) const { return _data + offset(x, y, z); }

	/// convertion to pointer of T* operators. useful for matvecop functions.
	inline operator T * () { return _data; }
	inline operator const T* () const { return _data; }

	/// index access operators
	inline const T& operator [] (int i) const { return _data[i]; }
	inline T& operator [] (int i) { return _data[i]; }

	/// tensor shape stuff
	inline T_SIZE width() const { return _width; }
	inline T_SIZE height() const { return _height; }
	inline T_SIZE depth() const { return _depth; }
	inline T_SIZE size() const { return _width * _height * _depth; }

	inline T_SIZE offset(T_SIZE x, T_SIZE y, T_SIZE z) const { return x + (y + z * _height) * _width; }
	inline T_SIZE offset(T_SIZE y, T_SIZE z) const { return (y + z * _height) * _width; }
	inline T_SIZE offset(T_SIZE z) const { return z * _width * _height; }

	/// iterator stuff
	inline iterator begin(T_SIZE stride) { return iterator(_data, stride); }
	inline iterator end(T_SIZE stride) { return begin(stride) + size(); }

	inline iterator begin(T_SIZE y, T_SIZE z, T_SIZE stride) { return iterator(data(y, z), stride); }
	inline iterator end(T_SIZE y, T_SIZE z, T_SIZE stride) { return begin(y, z, stride) + _width; }

	inline iterator begin(T_SIZE z, T_SIZE stride) { return iterator(data(z), stride); }
	inline iterator end(T_SIZE z, T_SIZE stride) { return begin(z, stride) + _width * _height; }

	inline range iter(T_SIZE stride) { return range(_data, stride, _width * _height * _depth); }
	inline range iter(T_SIZE y, T_SIZE z, T_SIZE stride) { return range(data(y, z), stride, _width); }
	inline range iter(T_SIZE z, T_SIZE stride) { return range(data(z), stride, _width * _height); }

	inline const_iterator begin(T_SIZE stride) const { return const_iterator(_data, stride); }
	inline const_iterator end(T_SIZE stride) const { return begin(stride) + size(); }

	inline const_iterator begin(T_SIZE y, T_SIZE z, T_SIZE stride) const { return const_iterator(data(y, z), stride); }
	inline const_iterator end(T_SIZE y, T_SIZE z, T_SIZE stride) const { return begin(y, z, stride) + _width; }

	inline const_iterator begin(T_SIZE z, T_SIZE stride) const { return const_iterator(data(z), stride); }
	inline const_iterator end(T_SIZE z, T_SIZE stride) const { return begin(z, stride) + _width * _height; }

	inline const_range iter(T_SIZE stride) const { return const_range(_data, stride, size()); }
	inline const_range iter(T_SIZE y, T_SIZE z, T_SIZE stride) const { return const_range(data(y, z), stride, _width); }
	inline const_range iter(T_SIZE z, T_SIZE stride) const { return const_range(data(z), stride, _width * _height); }

	/// Obtains a subtensor at depth z with a different depth.
	inline tensor<T, T_SIZE> window(T_SIZE z, T_SIZE depth) { return tensor<T, T_SIZE>(data(z), _width, _height, depth); }
	inline tensor<T, T_SIZE> window(T_SIZE z, T_SIZE depth) const { return tensor<T, T_SIZE>(data(z), _width, _height, depth); }

	/// Very dangerous and sometimes useful functions
	/// that essentially imply, that there are more than one tensor stacked one after another in memory.
	/// E.g. when we have a tensor(N,M,K) and have a subtensor of (N,M,L), where L < K and K is divisible by L.
	/// In that case we can have a window of depth=L and slide along depth K of the primary window.
	inline void operator >>= (T_SIZE v) {
		assert(_needs_free == false);
		_data += size() * v;
	}
	inline void operator <<= (T_SIZE v) {
		assert(_needs_free == false);
		_data -= size() * v;
	}

	/// indicator whether this tensor object owns the memory
	inline bool owns() const { return _needs_free; }
	inline void owns(bool val) { _needs_free = val; }

	/// allows to resize the tensor reallocating the memory if necessary
	inline void resize(T_SIZE width, T_SIZE height, T_SIZE depth) {
		if (_data != NULL && _needs_free) {
			delete[] _data;
		}
		if (width == 0 || height == 0 || depth == 0) {
			_width = _height = _depth = 0;
			_data = NULL;
		} else {
			_width = width;
			_height = height;
			_depth = depth;
			_data = new T[width * height * depth];
			_needs_free = true;
		}
	}
	inline void resize(const tensor<T, T_SIZE>& src) {
		resize(src.width(), src.height(), src.depth());
	}

	/// reshapes the tensor if resulting size is less or equal to the original size
	inline void reshape(T_SIZE width, T_SIZE height, T_SIZE depth) {
		assert(width * height * depth <= size());
		_width = width;
		_height = height;
		_depth = depth;
	}

	inline void reshape(const tensor<T, T_SIZE>& src) {
		reshape(src.width(), src.height(), src.depth());
	}

	/// clones this tensor and returns a tensor object, that does not own the memory.
	/// NOTE: this is dangerous, since memory is without ownership. owns must be explicitly set
	/// after call to clone.
	inline tensor<T, T_SIZE> clone(bool copy=true) const {
		tensor<T, T_SIZE> tmp(width(), height(), depth());
		if (copy)
			tmp.copy(*this);
		tmp.owns(false);
		return tmp;
	}

	/// creates a new tensor object from this one and returns a pointer to it.
	inline tensor<T, T_SIZE>* clone_new(bool copy=true) const {
		tensor<T, T_SIZE> *tmp = new tensor<T, T_SIZE>(width(), height(), depth());
		if (copy)
			tmp->copy(*this);
		return tmp;
	}

	/// fills the tensor with specific value.
	/// NOTE: for performance reasons filling with 0 will use memset.
	inline void fill(T val) {
		auto N = size();
		auto p = data();
		if (val == 0) {
			memset(p, sizeof(T) * N, 0);
			return;
		}
		while (N--)
			*(p++) = val;
	}

	inline void fill(T_SIZE z, T val) {
		auto N = width() * height();
		auto p = data(z);
		if (val == 0) {
			memset(p, sizeof(T) * N, 0);
			return;
		}
		while (N--)
			*(p++) = val;
	}

	inline void fill(T_SIZE y, T_SIZE z, T val) {
		auto N = width();
		auto p = data(y, z);
		if (val == 0) {
			memset(p, sizeof(T) * N, 0);
			return;
		}
		while (N--)
			*(p++) = val;
	}

	/// will call a function on each element. basically doing Ai = f(Ai)
	inline void map(std::function<T (T_SIZE idx, T val, void * params)> setter, void * params = NULL) {
		auto N = size();
		auto p = data();
		for (T_SIZE i = 0; i < N; i++) {
			*p = setter(i++, *p, params);
			++p;
		}
	}

	/// copy data from a pointer. size of data is equal to the size of this tensor
	inline void copy(const T * src, T_SIZE stride=1) {
		memcpy(data(), src, sizeof(T) * size());
	}

	/// copy data from a tensor. size of data copied is equal to the size of this tensor
	inline void copy(const tensor<T, T_SIZE>& src, T_SIZE stride=1) {
		assert(size() == src.size());

		copy(src.data(), stride);
	}

	/// will copy data from a tensor starting at depth z_offset.
	inline void copy_from(const tensor<T, T_SIZE>& src, T_SIZE z_offset) {
		assert(width() == src.width() && height() == src.height());
		assert((depth() + z_offset) <= src.depth());

		copy(src.data(z_offset));
	}
};


///
/// Implements a static tensor of preset size.
///
template<typename T, size_t width, size_t height, size_t depth, typename T_SIZE = ENN_DEFAULT_SIZE_TYPE>
class static_tensor : public tensor<T, T_SIZE> {
	T _arr[width * height * depth];
public:
	static_tensor() : tensor<T, T_SIZE>(&_arr, width, height, depth) {}
};

};

#endif
