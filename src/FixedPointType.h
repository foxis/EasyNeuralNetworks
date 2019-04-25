#if !defined(ENN_FIXED_POINT_TYPE_H)
#define ENN_FIXED_POINT_TYPE_H

#include <stdint.h>
#include <math.h>

namespace EasyNeuralNetworks {

template<typename T, int EXPONENT, bool SIGNED=true>
class FixedPointType {
	T raw;
	const T MSB_BIT = ~((~(T)0) >> 1);
	const T MSB_MASK = ~MSB_BIT;
	const T EXPONENT_MASK = ~(((~(T)0) >> EXPONENT) << EXPONENT);
public:
	FixedPointType(const FixedPointType<T, EXPONENT, SIGNED> &val) {
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

	#define ENN_CONVERSION_HELPER(TYPE) operator TYPE () const { return convert_to_int<TYPE>(raw); }

	ENN_CONVERSION_HELPER(int8_t)
	ENN_CONVERSION_HELPER(int16_t)
	ENN_CONVERSION_HELPER(int32_t)
	ENN_CONVERSION_HELPER(int64_t)
	ENN_CONVERSION_HELPER(uint8_t)
	ENN_CONVERSION_HELPER(uint16_t)
	ENN_CONVERSION_HELPER(uint32_t)
	ENN_CONVERSION_HELPER(uint64_t)

	#undef ENN_CONVERSION_HELPER

	operator float () const { return convert_to_float<float>(raw); }
	operator double () const { return convert_to_float<double>(raw); }

	#define ENN_ASSIGNMENT_HELPER(TYPE) void operator = (TYPE val) { raw = convert_from_int<TYPE>(val); }

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

	void operator +=(const FixedPointType<T, EXPONENT, SIGNED> &val) { raw += val.raw; }
	void operator -=(const FixedPointType<T, EXPONENT, SIGNED> &val) { raw -= val.raw; }
	void operator *=(const FixedPointType<T, EXPONENT, SIGNED> &val) { raw = mul(val.raw); }
	void operator /=(const FixedPointType<T, EXPONENT, SIGNED> &val) { raw = div(val.raw); }

	bool operator >(const FixedPointType<T, EXPONENT, SIGNED> &val) const { return raw > val.raw; }
	bool operator <(const FixedPointType<T, EXPONENT, SIGNED> &val) const { return raw < val.raw; }

	FixedPointType<T, EXPONENT, SIGNED>& operator +(const FixedPointType<T, EXPONENT, SIGNED> &val) const { return raw + val.raw; }
	FixedPointType<T, EXPONENT, SIGNED>& operator -(const FixedPointType<T, EXPONENT, SIGNED> &val) const { return raw - val.raw; }
	FixedPointType<T, EXPONENT, SIGNED>& operator *(const FixedPointType<T, EXPONENT, SIGNED> &val) const { return mul(val.raw); }
	FixedPointType<T, EXPONENT, SIGNED>& operator /(const FixedPointType<T, EXPONENT, SIGNED> &val) const { return div(val.raw); }

private:
	template<typename To>
	To convert_to_int(T val) {
		if (SIGNED) {
			T sign = val & MSB_BIT;
			return sign ? -((val & MSB_MASK) >> EXPONENT) : ((val & MSB_MASK) >> EXPONENT);
		} else {
			return val >> EXPONENT;
		}
	}
	template<typename To>
	To convert_to_float(T val) {
		return ((To)val) / (To)(1 << EXPONENT);
	}

	template<typename From>
	T convert_from_int(From val) {
		const From FROM_MSB = ~((~(From)0) >> 1);
		const From FROM_MSB_MASK = ~FROM_MSB;
		if (SIGNED) {
			From sign = val & FROM_MSB;
			return ((val & FROM_MSB_MASK) << EXPONENT) | (sign ? MSB_BIT : 0);
		} else {
			raw = val << EXPONENT;
		}
	}

	template<typename From>
	T convert_from_float(From val) {
		return val * (1 << EXPONENT);
	}

	#define ENN_DIV_HELPER(ENN_TYPE1, ENN_TMP) inline ENN_TYPE1 div(ENN_TYPE1 val) const { return (T)(((ENN_TMP)raw << EXPONENT) / (ENN_TMP)val); }
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
