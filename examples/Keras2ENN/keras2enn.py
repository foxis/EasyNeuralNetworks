#

import argparse
import keras as K
from collections import namedtuple


Args = namedtuple("Args", "output namespace dropout flatten verbose")


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
        self.args = args

        if args.verbose:
            print("Encountered layer", keras_layer.__class__.__name__, "as", keras_layer.name)
            print("\tinput shape: ", keras_layer.input_shape)
            print("\toutput shape: ", keras_layer.output_shape)
            if keras_layer.weights:
                print("\tweights shape: ", keras_layer.weights[0].shape)
                print("\tbias shape: ", keras_layer.weights[1].shape)
            if hasattr(keras_layer, "padding"):
                print("\tpadding: ", keras_layer.padding)
            if hasattr(keras_layer, "filters"):
                print("\tfilters: ", keras_layer.filters)
            if hasattr(keras_layer, "strides"):
                print("\tstrides: ", keras_layer.strides)
            if hasattr(keras_layer, "pool_size"):
                print("\tpool_size: ", keras_layer.pool_size)
            if hasattr(keras_layer, "rate"):
                print("\trate: ", keras_layer.rate)
            print("")

        assert (any(isinstance(keras_layer, t) for t in LayerWrapper.TYPES))

    def write_weights(self, f):
        w = self.keras_layer.get_weights()
        try:
            res = getattr(self, "w_" + self.keras_layer.__class__.__name__)(f, w)
            if res is not None:
                print("Writing weights for", self.name)

                if self.args.verbose:
                    print("\nweights: ", w[0].tolist())
                    print("\nbiases: ", w[1].tolist())
                    print("")

                wb, width, height, depth = res
                f.write("TYPE arr_{}[] PROGMEM = {{\n".format(self.name))
                for i, w in enumerate(wb):
                    if i % width == 0 and i != 0:
                        f.write("\n")
                    f.write("{}, ".format(w))
                if len(wb) % 20:
                    f.write("\n")
                f.write("}};\ntensor<TYPE> w_{0}(ProgmemHelper<TYPE>(arr_{0}), /* width= */ {1}, /* height= */ {2}, /* depth= */ {3});\n\n".format(self.name, width, height, depth))
        except KeyError:
            pass

    def write_layers(self, f, last):
        print("Writing layers for", self.name)
        if not last:
            shape = self.keras_layer.input_shape[1:]
            f.write("InputLayer<TYPE> input({});\n".format(", ".join("/* {} */ {}".format(n, i) for n, i in zip("width height depth".split(), shape))))
        try:
            getattr(self, "a_" + self.keras_layer.__class__.__name__)(f)
        except KeyError:
            pass

        f.write("{}<TYPE> {}(".format(self.enn_class, self.name))
        if last:
            f.write(last.name)
        else:
            f.write("input")
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

        width, channels, kernels = w[0].shape

        wb = []
        for w, b in zip(weights, bias):
            for c in w:
                wb += c + [b]

        return wb, (width * channels + 1), 1, kernels

    def w_Conv2D(self, f, w):
        weights = w[0].transpose().tolist()
        bias = w[1].tolist()

        width, height, channels, kernels = w[0].shape

        wb = []
        for w, b in zip(weights, bias):
            for c in w:
                for ww in zip(*c):
                    wb += ww
            wb += [b]

        return wb, (width * height * channels + 1), 1, kernels

    def w_Dense(self, f, w):
        weights = w[0].transpose().tolist()
        bias = w[1].tolist()

        wb = []
        for w, b in zip(weights, bias):
            wb += w + [b]

        return wb, len(weights[0]) + 1, len(bias), 1

    def l_Conv1D(self, f, l):
        kernel_w, kernel_h = self.keras_layer.kernel_size
        kernels = self.keras_layer.filters
        stride = self.keras_layer.strides[0]
        assert (self.keras_layer.strides[1] == stride)
        assert (self.keras_layer.padding == 'valid')

        f.write(", /* kernel_width= */ {}, /* kernels= */ {}, /* stride= */ {}, /* weights= */ w_{}, /* activation= */ act_{}".format(kernel_w, kernels, stride, self.name, self.name))

    def l_Conv2D(self, f, l):
        kernel_w, kernel_h = self.keras_layer.kernel_size
        kernels = self.keras_layer.filters
        stride = self.keras_layer.strides[0]
        assert (self.keras_layer.strides[1] == stride)
        assert (self.keras_layer.padding == 'valid')

        f.write(", /* kernel_width= */ {}, /* kernel_height= */ {}, /* kernels= */ {}, /* stride= */ {}, /* weights= */ w_{}, /* activation= */ act_{}".format(kernel_w, kernel_h, kernels, stride, self.name, self.name))

    def l_Dense(self, f, l):
        outputs = self.keras_layer.output_shape[-1]

        f.write(", /* out_width= */ {}, /* weights= */ w_{}, /* activation= */ act_{}".format(outputs, self.name, self.name))

    def l_Dropout(self, f, l):
        rate = self.keras_layer.rate
        f.write(", /* rate= */ {}".format(rate))

    def l_Dropout1D(self, f, l):
        rate = self.keras_layer.rate
        f.write(", /* rate= */ {}".format(rate))

    def l_Dropout2D(self, f, l):
        rate = self.keras_layer.rate
        f.write(", /* rate= */ {}".format(rate))

    def l_MaxPooling1D(self, f, l):
        kernel_w = self.keras_layer.pool_size[0]
        f.write(", /* width= */ {}, /* stride= */ {}".format(kernel_w, self.keras_layer.strides[0]))

    def l_MaxPooling2D(self, f, l):
        assert(self.keras_layer.strides[0] == self.keras_layer.strides[1])
        kernel_w, kernel_h = self.keras_layer.pool_size
        f.write(", /* width= */ {}, /* height= */ {}, /* stride= */ {}".format(kernel_w, kernel_h, self.keras_layer.strides[0]))

    def l_Flatten(self, f, l):
        pass

    def a_Dense(self, f):
        f.write("auto act_{} = {};\n".format(self.name, self.enn_activation))

    def a_Conv1D(self, f):
        f.write("auto act_{} = {};\n".format(self.name, self.enn_activation))

    def a_Conv2D(self, f):
        f.write("auto act_{} = {};\n".format(self.name, self.enn_activation))


def export_model_to_header(model, args):
    if args.verbose:
        print("Model: ", model.__class__.__name__)
        print("Output header file: ", args.output)
        print("Output namespace: ", args.namespace)
        print("Output nn: nn")
        print("")

    assert (isinstance(model, K.models.Sequential))

    layers = [LayerWrapper(l, args) for l in model.layers]
    layers = [l for l in layers if l.active]

    with open(args.output, "w") as f:
        f.write("""// Automatically generated NN header using keras2enn.py
//
#if !defined(K2ENN_{0}_H)
#define K2ENN_{0}_H

#include <NeuralNetwork.h>

namespace {1} {{

using namespace EasyNeuralNetworks;

// Neural network weights definition
""".format(args.output.split(".")[0].upper(), args.namespace))

        for l in layers:
            l.write_weights(f)

        f.write("\n\n// Neural network layers definition\n")

        last = None
        for l in layers:
            l.write_layers(f, last)
            last = l

        f.write("""// Neural network
NeuralNetwork<TYPE> nn({}, &input, {});

}};

#endif
""".format(sum(l.active for l in layers) + 1, ", ".join("&" + l.name for l in layers if l.active)))


def main(args):
    model = K.models.load_model(args.model)
    export_model_to_header(model, args)


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
