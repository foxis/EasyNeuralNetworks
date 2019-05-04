#include <Arduino.h>
#include <NeuralNetwork.h>

#define TYPE float

#include <xor.h>

using namespace EasyNeuralNetworks;


void setup() {
	Serial.begin(115200);
	Serial.println("Testing XOR NN with 2 hidden neurons...");
}

TYPE inputs[] = {
	0,0,
	1,0,
	0,1,
	1,1,
};

void loop() {
	TYPE o[4];
	unsigned long now = micros();
	tensor<TYPE> &input = XOR::nn.input();
	tensor<TYPE> &output = XOR::nn.output();

	for (int i = 0; i < 4; i++) {
		input.copy(inputs + i * 2);
		XOR::nn.calculate();
		o[i] = output[0];

		/*tensor<TYPE> &input = XOR::dense_1.inputs();
		for (int j = 0; j < input.size(); j++) {
			Serial.print(input[j]); Serial.print(" ");
		}
		Serial.println();

		tensor<TYPE> &hidden = XOR::dense_1.outputs();
		for (int j = 0; j < hidden.size(); j++) {
			Serial.print(hidden[j]); Serial.print(" ");
		}
		Serial.println();*/
	}

	now = micros() - now;

	Serial.print("Computed in: "); Serial.print(now); Serial.println("us");
	for (int i = 0; i < 4; i++) {
		Serial.print("Inputs: ");
		Serial.print(inputs[i * 2]); Serial.print(", "); Serial.print(inputs[i * 2 + 1]); Serial.println();
		Serial.print("Output: ");
		Serial.print((float)o[i]); Serial.println();
	}
	delay(1000);
}
