import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense

from keras2enn import Args, export_model_to_header

# the four different states of the XOR gate
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
training_data1 = np.array([[0, 0]], "float32")
training_data2 = np.array([[0, 1]], "float32")
training_data3 = np.array([[1, 0]], "float32")
training_data4 = np.array([[1, 1]], "float32")


# the four expected results in the same order
target_data = np.array([[-1], [1], [1], [-1]], "float32")

model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['binary_crossentropy'])

model.fit(training_data, target_data, nb_epoch=5000, verbose=2)

print (model.predict(training_data))

export_model_to_header(model, Args("xor.h", "XOR", False, False, True))


layer_name = 'dense_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

print(intermediate_layer_model.predict(training_data1))
print(intermediate_layer_model.predict(training_data2))
print(intermediate_layer_model.predict(training_data3))
print(intermediate_layer_model.predict(training_data4))

from datetime import datetime, timedelta
import json
import requests
import sys


