#if !defined(ENN_FIXED_POINT_TYPE_H)
#define ENN_FIXED_POINT_TYPE_H

#include <stdint.h>
#include <math.h>

namespace EasyNeuralNetworks {

template<typename T, int EXPONENT>
class FixedPointType {
	T raw;
public:
	FixedPointType() { raw = 0;	}
	FixedPointType(const FixedPointType<T, EXPONENT> &val) {
		this->raw = val.raw;
	}
	FixedPointType(int8_t val) { *this = val;	}
	FixedPointType(int16_t val) { *this = val; }
	FixedPointType(int32_t val) { *this = val; }
	FixedPointType(int64_t val) {	*this = val; }
	FixedPointType(uint8_t val) {	*this = val; }
	FixedPointType(uint16_t val) { *this = val; }
	FixedPointType(uint32_t val) { *this = val; }
	FixedPointType(uint64_t val) { *this = val; }
	FixedPointType(float val) { *this = val; }
	FixedPointType(double val) { *this = val; }

	#define ENN_CONVERSION_HELPER(TYPE) explicit operator TYPE () const { return raw >> EXPONENT; }

	ENN_CONVERSION_HELPER(int8_t)
	ENN_CONVERSION_HELPER(int16_t)
	ENN_CONVERSION_HELPER(int32_t)
	ENN_CONVERSION_HELPER(int64_t)
	ENN_CONVERSION_HELPER(uint8_t)
	ENN_CONVERSION_HELPER(uint16_t)
	ENN_CONVERSION_HELPER(uint32_t)
	ENN_CONVERSION_HELPER(uint64_t)

	#undef ENN_CONVERSION_HELPER

	explicit operator float () const { return convert_to_float<float>(raw); }
	explicit operator double () const { return convert_to_float<double>(raw); }

	#define ENN_ASSIGNMENT_HELPER(TYPE) void operator = (TYPE val) { raw = ((T)val) << EXPONENT; }

	ENN_ASSIGNMENT_HELPER(int8_t)
	ENN_ASSIGNMENT_HELPER(int16_t)
	ENN_ASSIGNMENT_HELPER(int32_t)
	ENN_ASSIGNMENT_HELPER(int64_t)
	ENN_ASSIGNMENT_HELPER(uint8_t)
	ENN_ASSIGNMENT_HELPER(uint16_t)
	ENN_ASSIGNMENT_HELPER(uint32_t)
	ENN_ASSIGNMENT_HELPER(uint64_t)

	#undef ENN_ASSIGNMENT_HELPER

	void operator =(float val) { raw = convert_from_float<float>(val); }
	void operator =(double val) { raw = convert_from_float<double>(val); }

	void operator +=(const FixedPointType<T, EXPONENT> &val) { raw += val.raw; }
	void operator -=(const FixedPointType<T, EXPONENT> &val) { raw -= val.raw; }
	void operator *=(const FixedPointType<T, EXPONENT> &val) { raw = mul(val.raw); }
	void operator /=(const FixedPointType<T, EXPONENT> &val) { raw = div(val.raw); }

	bool operator >(const FixedPointType<T, EXPONENT> &val) const { return raw > val.raw; }
	bool operator <(const FixedPointType<T, EXPONENT> &val) const { return raw < val.raw; }

	FixedPointType<T, EXPONENT> operator +(const FixedPointType<T, EXPONENT> &val) const { FixedPointType<T, EXPONENT> tmp; tmp.raw = raw + val.raw; return tmp; }
	FixedPointType<T, EXPONENT> operator -(const FixedPointType<T, EXPONENT> &val) const { FixedPointType<T, EXPONENT> tmp; tmp.raw = raw - val.raw; return tmp; }
	FixedPointType<T, EXPONENT> operator *(const FixedPointType<T, EXPONENT> &val) const { FixedPointType<T, EXPONENT> tmp; tmp.raw = mul(val.raw); return tmp; }
	FixedPointType<T, EXPONENT> operator /(const FixedPointType<T, EXPONENT> &val) const { FixedPointType<T, EXPONENT> tmp; tmp.raw = div(val.raw); return tmp; }

	FixedPointType<T, EXPONENT>& operator ++() {
		raw += ((T)1) << EXPONENT;
		return *this;
	}
	FixedPointType<T, EXPONENT> operator ++(int) {
		FixedPointType<T, EXPONENT> tmp = *this;
		++*this;
		return tmp;
	}

	FixedPointType<T, EXPONENT>& operator --() {
		raw -= ((T)1) << EXPONENT;
		return *this;
	}
	FixedPointType<T, EXPONENT> operator --(int) {
		FixedPointType<T, EXPONENT> tmp = *this;
		--*this;
		return tmp;
	}
private:

	template<typename To>
	To convert_to_float(T val) const {
		return ((To)val) / (To)(((T)1) << EXPONENT);
	}

	template<typename From>
	T convert_from_float(From val) const {
		return val * (((T)1) << EXPONENT);
	}

	#define ENN_DIV_HELPER(ENN_TYPE1, ENN_TMP) inline ENN_TYPE1 div(ENN_TYPE1 val) const { return (T)((((ENN_TMP)raw) << EXPONENT) / (ENN_TMP)val); }
	#define ENN_MUL_HELPER(ENN_TYPE1, ENN_TMP) inline ENN_TYPE1 mul(ENN_TYPE1 val) const { return (T)(((ENN_TMP)raw * (ENN_TMP)val) >> EXPONENT); }

	ENN_MUL_HELPER(int8_t, int16_t);
	ENN_MUL_HELPER(int16_t, int32_t);
	ENN_MUL_HELPER(int32_t, int64_t);
	ENN_MUL_HELPER(int64_t, int64_t);
	ENN_MUL_HELPER(uint8_t, uint16_t);
	ENN_MUL_HELPER(uint16_t, uint32_t);
	ENN_MUL_HELPER(uint32_t, uint64_t);
	ENN_MUL_HELPER(uint64_t, uint64_t);

	ENN_DIV_HELPER(int8_t, int16_t);
	ENN_DIV_HELPER(int16_t, int32_t);
	ENN_DIV_HELPER(int32_t, int64_t);
	ENN_DIV_HELPER(int64_t, int64_t);
	ENN_DIV_HELPER(uint8_t, uint16_t);
	ENN_DIV_HELPER(uint16_t, uint32_t);
	ENN_DIV_HELPER(uint32_t, uint64_t);
	ENN_DIV_HELPER(uint64_t, uint64_t);

	#undef ENN_MUL_HELPER
	#undef ENN_DIV_HELPER
};

};

#endif
