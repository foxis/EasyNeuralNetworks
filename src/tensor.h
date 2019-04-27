#if !defined(ENN_TENSOR_H)
#define ENN_TENSOR_H

#include <iterator>
#include <assert.h>

namespace EasyNeuralNetworks {

template<typename T, typename T_SIZE>
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
		~iterator_() {}

		inline iterator_<IS_CONST>  operator++(int) /* postfix */         { iterator_<IS_CONST> tmp(ptr, stride); ptr += stride; return tmp; }
    inline iterator_<IS_CONST>& operator++()    /* prefix */          { ptr += stride; return *this; }
    inline T_TYPE_REF& operator* () const                    { return *ptr; }
    inline T_TYPE   operator->() const                    { return ptr; }
		inline iterator_<IS_CONST>  operator+ (T_SIZE v)   const { return ptr + v; }
		inline void operator >>= (T_SIZE v) { ptr += v; }

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
		inline iterator_<true>& begin() const { return _begin; }
		inline iterator_<true>& end() const { return _end; }
		inline iterator_<false>& begin() { return _begin; }
		inline iterator_<false>& end() { return _end; }

		inline void operator += (T_SIZE v) { _begin += v; _end += v; }

		inline void operator >>= (T_SIZE v) { _begin >>= v; _end >>= v; }
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

	~tensor() {
		if (_needs_free && _data != NULL)
			delete[] _data;
	}

	// access stuff
	inline T * data() { return _data; }
	inline const T* data() const { return _data; }

	inline T operator [] (T_SIZE i) const { return _data[i]; }
	inline T& operator [] (T_SIZE i) { return _data[i]; }

	// shape stuff
	inline T_SIZE width() const { return _width; }
	inline T_SIZE height() const { return _height; }
	inline T_SIZE depth() const { return _depth; }
	inline T_SIZE size() const { return _width * _height * _depth; }

	inline T_SIZE offset(T_SIZE x, T_SIZE y, T_SIZE z) const { return x + y * _width + z * _width * _height; }

	// iterator stuff
	iterator& begin(T_SIZE stride) { return iterator(_data, stride); }
	iterator& end(T_SIZE stride) { return begin(stride) + _width * _height * _depth; }

	iterator& begin(T_SIZE y, T_SIZE z, T_SIZE stride) { return iterator(data + y * _width + z * _width * _height, stride); }
	iterator& end(T_SIZE y, T_SIZE z, T_SIZE stride) { return begin(y, z, stride) + _width; }

	iterator& begin(T_SIZE z, T_SIZE stride) { return iterator(_data, _width * _height * z, stride); }
	iterator& end(T_SIZE z, T_SIZE stride) { return end(z, stride) + _width * _height; }

	range& iter() { return range(data, 1, _width * _height * _depth); }
	range& iter(T_SIZE y, T_SIZE z) { return range(data + y * _width + z * _width * _height, 1, _width); }
	range& iter(T_SIZE z) { return range(data + z * _width * _height, 1, _width * _height); }

	const_iterator& begin(T_SIZE stride) const { return const_iterator(_data, stride); }
	const_iterator& end(T_SIZE stride) const { return begin(stride) + _width * _height * _depth; }

	const_iterator& begin(T_SIZE y, T_SIZE z, T_SIZE stride) const { return const_iterator(data + y * _width + z * _width * _height, stride); }
	const_iterator& end(T_SIZE y, T_SIZE z, T_SIZE stride) const { return begin(y, z, stride) + _width; }

	const_iterator& begin(T_SIZE z, T_SIZE stride) const { return const_iterator(_data, _width * _height * z, stride); }
	const_iterator& end(T_SIZE z, T_SIZE stride) const { return end(z, stride) + _width * _height; }

	const_range& iter() const { return const_range(data, 1, _width * _height * _depth); }
	const_range& iter(T_SIZE y, T_SIZE z) const { return const_range(data + y * _width + z * _width * _height, 1, _width); }
	const_range& iter(T_SIZE z) const { return const_range(data + z * _width * _height, 1, _width * _height); }

	tensor& window(T_SIZE z) { return tensor(data + z * _width * _height, width, height, 1); }
	tensor& window(T_SIZE y, T_SIZE z) { return tensor(data + y * _width + z * _width * _height, _width, 1, 1); }

	const tensor& window(T_SIZE z) const { return tensor(data + z * _width * _height, width, height, 1); }
	const tensor& window(T_SIZE y, T_SIZE z) const { return tensor(data + y * _width + z * _width * _height, _width, 1, 1); }
};

};

#endif
