#include <Arduino.h>
#include <NeuralNetwork.h>
#include <BackPropTrainer.h>
using namespace EasyNeuralNetworks;

//#define TYPE FixedPointType<int32_t, 16>
#define TYPE float

//TanhActivation<TYPE> activation;
SigmoidActivation<TYPE> activation;
//SoftplusActivation<TYPE> activation1;
ReLUActivation<TYPE> activation1(.001);

InputLayer<TYPE> input(2);
DenseLayer<TYPE> hidden(input, 5, activation1);
DenseLayer<TYPE> output(hidden, 1, activation);

NeuralNetwork<TYPE> nn(3, &input, &hidden, &output);

BackPropTrainer<TYPE, ENN_WEIGHTS_FLAT> trainer(2.0f, .001f, L2Loss<TYPE>(), [](TYPE error, size_t epoch, void * data) {
	if (epoch % 100 == 0) {
		Serial.print("Epoch ");
		Serial.print(epoch);
		Serial.print(", error ");
		Serial.println((float)error);
	}
	delay(1);
	return error > 0.001;
});

TYPE inputs[] = {
	0,0,
	1,0,
	0,1,
	1,1,
};
TYPE outputs[] = {
	0,1,1,0
};

tensor<TYPE> input_tensor(inputs, 2, 1, 4);
tensor<TYPE> output_tensor(outputs, 1, 1, 4);

void setup() {
	Serial.begin(115200);
	Serial.println("Testing Training XOR NN with 3 hidden neurons...");

	unsigned long now = micros();
	nn.train(input_tensor, output_tensor, trainer, 5000);
	now = micros() - now;
	Serial.print("Trained in: "); Serial.print(now); Serial.println("us");
}

void loop() {
	TYPE o[4];
	unsigned long now = micros();
	TYPE * p = inputs;

	for (int i = 0; i < 4; i++) {
		input.inputs().copy(input_tensor.window(i));
		nn.calculate();
		o[i] = output.outputs()[0];
		p += 2;
	}

	now = micros() - now;

	Serial.print("Computed in: "); Serial.print(now); Serial.println("us");
	for (int i = 0; i < 4; i++) {
		Serial.print("Inputs: ");
		Serial.print((float)inputs[i * 2]); Serial.print(", "); Serial.print((float)inputs[i * 2 + 1]); Serial.println();
		Serial.print("Output: ");
		Serial.print((float)o[i]); Serial.println();
	}

	delay(1000);
}
