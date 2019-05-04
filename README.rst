EasyNeuralNetworks
==================

NeuralNetwork framework mainly for AVR/ESP microcontrollers, but can be used on other architectures.

Supported Layers
----------------

* Fully Connected (DenseLayer)
* Convolution (ConvLayer1D and ConvLayer2D)
* Max Pooling (MaxPoolingLayer1D and MaxPoolingLayer1D)
* Zero Padding (ZeroPaddingLayer1D and ZeroPaddingLayer2D)
* Reshaping Layer (ReshapeLayer)
* Flattening Layer (FlattenLayer)
* Drop Out Layer (DropOutLayer, DropOutLayer1D and DropOutLayer2D)
* Concatenation layer (ConcatLayer)

Supported trainers:
	Gradient Descent Back Propagation for Dense and Convolution layers.
