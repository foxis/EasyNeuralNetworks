import argparse
import keras as K


class LayerWrapper(object):
	TYPES = [
		K.layers.Conv1D,
		K.layers.Conv2D,
		K.layers.Dense,
		K.layers.DropOut,
		K.layers.Flatten,
		K.layers.MaxPooling1D,
		K.layers.MaxPooling2D]

	def __init__(self, keras_layer, index, args):
		self.keras_layer = keras_layer
		self.index = index
		self.use_dropout = args.dropout
		self.use_flatten = args.flatten

		assert(any(isinstance(l, t) for t in LayerWrapper.TYPES))

	def write_weights(self, f):
		w = self.keras_layer.weights
		kernel = w[0];
		bias = w[1];

	def write_layers(self, f, last):
		pass

	@property
	def name(self):
		self.keras_layer.name

	@property
	def active(self):
		if isinstance(self.keras_layer, LayerWrapper.TYPES[3]):
			return self.use_dropout
		elif isinstance(self.keras_layer, LayerWrapper.TYPES[4]):
			return self.use_flatten
		else:
			return True


def main(args):
	model = K.models.load_model(args.model)
	assert(isinstance(model, K.models.Sequential))

	layers = [LayerWrapper(l, args) for l in model.layers]

	with open(args.output, "w") as f:
		f.write("""// Automatically generated NN header using keras2enn.py
//
#if !defined(K2ENN_{0}_H)
#define K2ENN_{0}_H

#include <NeuralNetwork.h>

using namespace EasyNeuralNetworks;

namespace {1} {

// Neural network weights definition
""".format(args.output.split(".").upper(), args.namespace))

		for l in layers:
			l.write_weights(f)

		f.write("""

// Neural network layers definition
""")

		last = None;
		for l in layers:
			l.write_layers(f, last)
			last = l

		f.write("""
// Neural network
NeuralNetwork nn({}, {});

};

#endif
""".format(sum(l.active for l in layers), ", ".join(l.name for l in layers if l.active)))

#endif
""")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="MQTT Client for Weather Station readings monitoring.")
	parser.add_argument("model", help="Keras model file")
	parser.add_argument("output", help="Easy Neural Networks NN definition header file")
	parser.add_argument("namespace", help="Namespace to place the code under. (default None)")
	parser.add_argument("--dropout", const=True, default=False, action='store_const', help="Include DropOut layers (default False)")
	parser.add_argument("--flatten", const=True, default=False, action='store_const', help="Include Flatten layers (default False)")
    parser.add_argument("-v", "--verbose", const=True, default=False, action='store_const', help="Verbose output to stdout")
    args = parser.parse_args()

	main(args)
