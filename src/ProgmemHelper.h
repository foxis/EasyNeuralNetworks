#if !defined(ENN_PROGMEM_HELPER_H)
#define ENN_PROGMEM_HELPER_H

#include <pgmspace.h>

namespace EasyNeuralNetworks {

template <typename T>
class ProgmemHelper {
	const void * _flash;
public:
	ProgmemHelper(const void * flash) {
		_flash = flash;
	}

	void read(T * dst, size_t items) const {
		memcpy_P(dst, _flash, sizeof(T) * items);
	}

	T * read(size_t items) const {
		T * buf = (T*)malloc(sizeof(T) * items);
		read(buf, items);
		return buf;
	}
};

};

#endif
