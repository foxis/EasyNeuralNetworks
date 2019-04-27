#include <Arduino.h>
#include <NeuralNetwork.h>

using namespace EasyNeuralNetworks;

#define TYPE float

TanhActivation<TYPE> tanha;

///
/// Weights from: https://towardsdatascience.com/tflearn-soving-xor-with-a-2x2x1-feed-forward-neural-network-6c07d88689ed
///
TYPE weights[] PROGMEM = {
	3.86708593, 3.87053323, -1.82562542,
  -3.11288071, -3.1126008, 4.58438063
};
TYPE weights1[] PROGMEM = {
	5.19325304, 5.19325304, -4.87336922
};

tensor<TYPE> hidden_weights(ProgmemHelper<TYPE>(weights), 3, 2, 1);
tensor<TYPE> output_weights(ProgmemHelper<TYPE>(weights1), 3, 1, 1);

InputLayer<TYPE> input(2);
DenseLayer<TYPE> hidden(input, 2, hidden_weights, tanha);
DenseLayer<TYPE> output(hidden, 1, output_weights, tanha);

NeuralNetwork<TYPE> nn(3, &input, &hidden, &output);

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
	float o[4];
	unsigned long now = micros();

	for (int i = 0; i < 4; i++) {
		input.inputs().copy(inputs + i * 2);
		nn.calculate();
		o[i] = output.outputs()[0];
	}

	now = micros() - now;

	Serial.print("Computed in: "); Serial.print(now); Serial.println("us");
	for (int i = 0; i < 4; i++) {
		Serial.print("Inputs: ");
		Serial.print(inputs[i * 2]); Serial.print(", "); Serial.print(inputs[i * 2 + 1]); Serial.println();
		Serial.print("Output: ");
		Serial.print(o[i]); Serial.println();
	}
	delay(1000);
}
