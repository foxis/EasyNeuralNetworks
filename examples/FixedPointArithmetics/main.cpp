#include <Arduino.h>
#include <NeuralNetwork.h>
#include <BackPropTrainer.h>
using namespace EasyNeuralNetworks;

void setup() {
	Serial.begin(115200);
	Serial.println("Testing Fixed point arithmetics...");
}

template<typename T, typename T1, int EXPONENT>
void test_conversion(T initial, float add=78, float mul=31, float div=15) {
	T result;
	FixedPointType<T1, EXPONENT> test;

	Serial.print("Testing conversion "); Serial.print(__PRETTY_FUNCTION__ );
	Serial.println();

	test = initial;
	result = (T)test;

	Serial.print("initial: "); Serial.println((float)initial);
	Serial.print("result: "); Serial.println((float)result);
	Serial.print("test: "); Serial.println((float)test);

	Serial.print("add: "); Serial.print((float)(test + add)); Serial.print(" "); Serial.println((float)(initial) + add);
	Serial.print("add: "); Serial.print((float)(test + (1/add))); Serial.print(" "); Serial.println((float)(initial) + (1/add));
	Serial.print("sub: "); Serial.print((float)(test - add)); Serial.print(" "); Serial.println((float)(initial) - add);
	Serial.print("sub: "); Serial.print((float)(test - (1/add))); Serial.print(" "); Serial.println((float)(initial) - (1/add));

	Serial.print("mul: "); Serial.print((float)(test * mul));Serial.print(" ");  Serial.println((float)(initial) * mul);
	Serial.print("mul: "); Serial.print((float)(test * -mul)); Serial.print(" "); Serial.println((float)(initial) * -mul);
	Serial.print("mul: "); Serial.print((float)(test * (1.0f / mul))); Serial.print(" "); Serial.println((float)(initial) * (1/mul));

	Serial.print("div: "); Serial.print((float)(test / div)); Serial.print(" "); Serial.println((float)(initial) / div);
	Serial.print("div: "); Serial.print((float)(test / -div)); Serial.print(" "); Serial.println((float)(initial) / -div);
	Serial.print("div: "); Serial.print((float)(test / (1.0f / div))); Serial.print(" "); Serial.println((float)(initial) / (1/div));

	Serial.print("test++: "); Serial.print((float)test); Serial.print(" "); Serial.println((float)(test ++));
	Serial.print("test--: "); Serial.print((float)test); Serial.print(" "); Serial.println((float)(test --));
	Serial.print("++test: "); Serial.print((float)test); Serial.print(" "); Serial.println((float)(++test));
	Serial.print("--test: "); Serial.print((float)test); Serial.print(" "); Serial.println((float)(--test));

}


void loop() {
	FixedPointType<int32_t, 16> test = 75;

	test_conversion<int8_t, int32_t, 8>(75);
	test_conversion<int8_t, int32_t, 8>(-76);
	test_conversion<int16_t, int32_t, 8>(775);
	test_conversion<int16_t, int32_t, 8>(-175);

	test_conversion<int8_t, uint32_t, 16>(75);
	test_conversion<int8_t, uint32_t, 16>(-76);
	test_conversion<int16_t, uint32_t, 8>(775);
	test_conversion<int16_t, uint32_t, 8>(-175);

	test_conversion<uint8_t, int32_t, 16>(75);
	test_conversion<uint8_t, int32_t, 8>(175);

	test_conversion<float, int32_t, 20>(7.5);
	test_conversion<double, int32_t, 20>(-7.6);

	test_conversion<FixedPointType<int32_t, 16>, int32_t, 16>(test);

	while (true) delay(1000);
}
