#if !defined(ENN_PROGMEM_HELPER_H)
#define ENN_PROGMEM_HELPER_H

#include <stdlib.h>
#include <pgmspace.h>

namespace EasyNeuralNetworks {

typedef std::function<void (void * dst, const void * src, size_t cb)> Reader_t;

///
/// Simple Progmem data reader used for e.g. loading weights from flash memory
///
template <typename T>
class ProgmemHelper {
public:
	ProgmemHelper(const void * flash, Reader_t reader = memcpy_P) {
		_flash = flash;
		_reader = reader;
	}

	virtual void read(T * dst, size_t items) const {
		_reader(dst, _flash, sizeof(T) * items);
	}

	T * read(size_t items) const {
		T * buf = (T*)malloc(sizeof(T) * items);
		read(buf, items);
		return buf;
	}

protected:
	const void * _flash;
	Reader_t _reader;
};

///
/// Type Casting Progmem data reader
/// Used for situations where weights are stored as floats
/// and the user wants to cast them to e.g. FixedPointType
///
template <typename T, typename T_SOURCE>
class CastProgmemHelper : public ProgmemHelper<T> {
public:
		CastProgmemHelper(const void * flash, Reader_t reader = memcpy_P)
			: ProgmemHelper<T>(flash, reader) {}

	virtual void read(T * dst, size_t items) const {
		T_SOURCE * tmp = (T_SOURCE*)malloc(sizeof(T_SOURCE) * items);
		T_SOURCE *p = tmp;
		this->_reader(dst, this->_flash, sizeof(T_SOURCE) * items);
		for (size_t i = 0; i < items; i++) {
			*dst = *p;
			++p;
			++dst;
		}
		free(tmp);
	}

};

};

#endif
