#include <Arduino.h>
#include <NeuralNetwork.h>
#include <BackPropTrainer.h>
using namespace EasyNeuralNetworks;

SigmoidActivation<float> sigma;

InputLayer<float> input(2);
DenseLayer<float> hidden(input, 3, sigma);
DenseLayer<float> output(hidden, 1, sigma);

NeuralNetwork<float> nn(3, &input, &hidden, &output);

BackPropTrainer<float> trainer(0.5, .001);

float inputs[] = {
	0,0,
	1,0,
	0,1,
	1,1,
};
float outputs[] = {
	0,1,1,0
};

void setup() {
	Serial.begin(115200);
	Serial.println("Testing Training XOR NN with 3 hidden neurons...");

	unsigned long now = micros();
	nn.train((float*)inputs, (float*)outputs, 4, trainer, 1000);
	now = micros() - now;
	Serial.print("Trained in: "); Serial.print(now); Serial.println("us");
}

void loop() {
	float o[4];
	unsigned long now = micros();
	float * p = inputs;

	for (int i = 0; i < 4; i++) {
		input.inputs()[0] = p[0];
		input.inputs()[1] = p[1];
		nn.calculate();
		o[i] = output.outputs()[0];
		p += 2;
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
