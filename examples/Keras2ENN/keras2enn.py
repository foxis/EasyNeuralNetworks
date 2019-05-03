#

import argparse
import keras as K


class LayerWrapper(object):
    TYPES = [
        K.layers.Conv1D,
        K.layers.Conv2D,
        K.layers.Dense,
        K.layers.Dropout,
        #K.layers.Dropout1D,
        #K.layers.Dropout2D,
        K.layers.Flatten,
        K.layers.MaxPooling1D,
        K.layers.MaxPooling2D]

    NAMES = [
        "ConvLayer1D",
        "ConvLayer2D",
        "DenseLayer",
        "DropoutLayer",
        "FlattenLayer",
        "MaxPoolingLayer1D",
        "MaxPoolingLayer2D",
    ]

    CONVERSION = dict(zip(TYPES, NAMES))

    ACTIVATIONS = {
        "relu": "ReLUActivation<TYPE>()",
        "softmax": "SoftmaxActivation<TYPE>()",
        "softplus": "SoftplusActivation<TYPE>()",
        "sigmoid": "SigmoidActivation<TYPE>()",
        "tanh": "TanhActivation<TYPE>()",
    }

    def __init__(self, keras_layer, args):
        self.keras_layer = keras_layer
        self.use_dropout = args.dropout
        self.use_flatten = args.flatten

        assert (any(isinstance(keras_layer, t) for t in LayerWrapper.TYPES))

    def write_weights(self, f):
        w = self.keras_layer.get_weights()
        print("Writing weights for", self.name)
        f.write("TYPE {}[] PROGMEM = {{\r\n".format(self.name))
        wb = getattr(self, "w_" + self.keras_layer.__class__.__name__)(f, w)
        for i, w in enumerate(wb):
            f.write("{}, ".format(w))
            if i % 20 == 0 and i != 0:
                f.write("\n")
        if len(wb) % 20:
            f.write("\n")
        f.write("};\n\n")

    def write_layers(self, f, last):
        print("Writing layers for", self.name)
        f.write("{}<TYPE> {}(".format(self.enn_class, self.name))
        if last:
            f.write(last.name)
        getattr(self, "l_" + self.keras_layer.__class__.__name__)(f, last)
        f.write(");\n")

    @property
    def name(self):
        return self.keras_layer.name

    @property
    def enn_class(self):
        return LayerWrapper.CONVERSION[self.keras_layer.__class__]

    @property
    def enn_activation(self):
        return LayerWrapper.ACTIVATIONS[self.keras_layer.activation.__name__]

    @property
    def active(self):
        if any(isinstance(self.keras_layer, t) for t in LayerWrapper.TYPES[3:4]):
            return self.use_dropout
        elif any(isinstance(self.keras_layer, t) for t in LayerWrapper.TYPES[4:5]):
            return self.use_flatten
        else:
            return True

    def w_Conv1D(self, f, w):
        weights = w[0].transpose().tolist()
        bias = w[1].tolist()

        wb = []
        for w, b in zip(weights, bias):
            wb += w + [b]

        return wb

    def w_Conv2D(self, f, w):
        weights = w[0].transpose().tolist()
        bias = w[1].tolist()

        wb = []
        for w, b in zip(weights, bias):
            for ww in zip(*w):
                wb += ww
            wb += b

        return wb

    def w_Dense(self, f, w):
        weights = w[0].transpose().tolist()
        bias = w[1].tolist()

        wb = []
        for w, b in zip(weights, bias):
            wb += w + [b]

        return wb

    def w_Dropout(self, f, w): pass
    def w_MaxPooling1D(self, f, w): pass
    def w_MaxPooling2D(self, f, w): pass
    def w_Flatten(self, f, w): pass

    def l_Conv1D(self, f, l):
        kernel_w, kernel_h = self.keras_layer.kernel_size
        kernels = self.keras_layer.filters
        stride = self.keras_layer.strides[0]
        assert (self.keras_layer.strides[1] == stride)
        assert (self.keras_layer.padding == 'valid')

        if not l:
            width = int(self.keras_layer.input.shape[1])
            assert (int(self.keras_layer.input.shape[1]) == 1)
            depth = int(self.keras_layer.input.shape[-1])
            f.write("{}, {}".format(width, depth))

        f.write(", ProgmemHelper<TYPE>(w_{}), {}, {}, {}, {}".format(self.name, kernel_w, kernels, stride, self.enn_activation))

    def l_Conv2D(self, f, l):
        kernel_w, kernel_h = self.keras_layer.kernel_size
        kernels = self.keras_layer.filters
        stride = self.keras_layer.strides[0]
        assert (self.keras_layer.strides[1] == stride)
        assert (self.keras_layer.padding == 'valid')

        if not l:
            width = int(self.keras_layer.input.shape[1])
            height = int(self.keras_layer.input.shape[2])
            depth = int(self.keras_layer.input.shape[-1])
            f.write("{}, {}, {}".format(width, height, depth))

        f.write(", ProgmemHelper<TYPE>(w_{}), {}, {}, {}, {}, {}".format(self.name, kernel_w, kernel_h, kernels, stride, self.enn_activation))

    def l_Dense(self, f, l):
        outputs = self.keras_layer.output_shape[-1]

        if not l:
            width = int(self.keras_layer.input.shape[1])
            height = int(self.keras_layer.input.shape[2])
            depth = int(self.keras_layer.input.shape[-1])
            f.write("{}, {}, {}".format(width, height, depth))

        f.write(", {}, ProgmemHelper<TYPE>(w_{}), {}".format(outputs, self.name, self.enn_activation))

    def l_Dropout(self, f, l):
        rate = self.keras_layer.rate
        f.write(", {}".format(rate))

    def l_Dropout1D(self, f, l):
        rate = self.keras_layer.rate
        f.write(", {}".format(rate))

    def l_Dropout2D(self, f, l):
        rate = self.keras_layer.rate
        f.write(", {}".format(rate))

    def l_MaxPooling1D(self, f, l):
        kernel_w = self.keras_layer.pool_size[0]
        f.write(", {}".format(kernel_w))

    def l_MaxPooling2D(self, f, l):
        kernel_w, kernel_h = self.keras_layer.pool_size
        f.write(", {}, {}".format(kernel_w, kernel_h))

    def l_Flatten(self, f, l): pass


def main(args):
    model = K.models.load_model(args.model)
    assert (isinstance(model, K.models.Sequential))

    layers = [LayerWrapper(l, args) for l in model.layers]
    layers = [l for l in layers if l.active]

    with open(args.output, "w") as f:
        f.write("""// Automatically generated NN header using keras2enn.py
//
#if !defined(K2ENN_{0}_H)
#define K2ENN_{0}_H

#include <NeuralNetwork.h>

using namespace EasyNeuralNetworks;

namespace {1} {{

// Neural network weights definition
""".format(args.output.split(".")[0].upper(), args.namespace))

        for l in layers:
            l.write_weights(f)

        f.write("""

// Neural network layers definition
""")

        last = None
        for l in layers:
            l.write_layers(f, last)
            last = l

        f.write("""
// Neural network
NeuralNetwork nn({}, {});

}};

#endif
""".format(sum(l.active for l in layers), ", ".join("&" + l.name for l in layers if l.active)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MQTT Client for Weather Station readings monitoring.")
    parser.add_argument("model", help="Keras model file")
    parser.add_argument("output", help="Easy Neural Networks NN definition header file")
    parser.add_argument("namespace", help="Namespace to place the code under. (default None)")
    parser.add_argument("--dropout", const=True, default=False, action='store_const',
                        help="Include DropOut layers (default False)")
    parser.add_argument("--flatten", const=True, default=False, action='store_const',
                        help="Include Flatten layers (default False)")
    parser.add_argument("-v", "--verbose", const=True, default=False, action='store_const', help="Verbose output to stdout")
    args = parser.parse_args()

    main(args)
