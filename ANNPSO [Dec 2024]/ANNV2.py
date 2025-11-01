## ANNV2.1


import tensorflow as tf
import numpy as np

##ann = tf.keras.models.Sequential()
##ann.add(tf.keras.layers.Dense(units=10,activation="relu"))
##ann.add(tf.keras.layers.Dense(units=10,activation="relu"))
##ann.add(tf.keras.layers.Dense(units=1))
##ann.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
##
##
##ann.fit(np.array([[0, 1], [1, 0], [1, 1], [0, 0]]), np.array([1, 1, 0, 0]), batch_size=32, epochs=1000)
##
##print(ann.predict(np.array([[0, 1]])))



class Network:
    def __init__(self, inputSize, layerSize, nLayers): # This has a rectangular geometry
        self.inputDim = inputSize
        self.nLayers = nLayers
        self.layerSize = layerSize

        self.model = tf.keras.models.Sequential()

        for _ in range(0, nLayers-1):
            self.model.add(tf.keras.layers.Dense(units=layerSize,activation="relu"))

        self.model.add(tf.keras.layers.Dense(units=1))
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])

    def train(self, x_train, y_train, learning_rate, epochs):
        self.model.fit(np.array(x_train), np.array(y_train), epochs=epochs)
        

    def predict(self, X):
        return np.array(self.model.predict(np.array(X)).flat)

    
