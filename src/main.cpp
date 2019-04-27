#include <Arduino.h>
#include <NeuralNetwork.h>
#include <BackPropTrainer.h>
using namespace EasyNeuralNetworks;

#define TYPE float

TYPE inputs1[] = {
	0,0,
	1,0,
	0,1,
	1,1,
};

TYPE inputs2[] = {
	2,3,
	4,5,
	6,7,
	8,9,
};

TYPE inputs3[] = {
	2,
	3,
	4,
	5,
};

TYPE inputs4[] PROGMEM = {
	12,
	13,
	14,
	15,
};

tensor<TYPE> tensor1(inputs1, 2, 1, 4);
tensor<TYPE> tensor2(inputs2, 2, 1, 4);
tensor<TYPE> tensor3(inputs3, 1, 1, 4);
tensor<TYPE> tensor4(ProgmemHelper<TYPE>(inputs4), 1, 1, 4);

void print_tensor_cr(const char* name, const tensor<TYPE>& t) {
	Serial.print("Const Tensor Ref "); Serial.print(name); Serial.print(" ");
	Serial.print(t.width()); Serial.print("x"); Serial.print(t.height()); Serial.print("x"); Serial.print(t.depth()); Serial.print(": ");
	for (auto &v : t.iter(1)) {
		Serial.print(v); Serial.print(" ");
	}
	Serial.println();
}
void print_tensor_r(const char* name, tensor<TYPE>& t) {
	Serial.print("Tensor Ref "); Serial.print(name); Serial.print(" ");
	for (auto &v : t.iter(1)) {
		Serial.print(v); Serial.print(" ");
	}
	Serial.println();
}
void print_tensor(const char* name, tensor<TYPE> t) {
	Serial.print("Tensor "); Serial.print(name); Serial.print(" ");
	for (auto &v : t.iter(1)) {
		Serial.print(v); Serial.print(" ");
	}
	Serial.println();
}

void setup() {
	Serial.begin(115200);
	Serial.println("Testing tensor class...");

	print_tensor_cr("1", tensor1);
	print_tensor_cr("2", tensor2);
	print_tensor_cr("3", tensor3);
	print_tensor_cr("4", tensor4);

	print_tensor_r("2", tensor2);
	print_tensor("2", tensor2);

	tensor<TYPE> copy = tensor2;
	print_tensor_cr("copy", copy);

}

void loop() {
	delay(1000);
}
