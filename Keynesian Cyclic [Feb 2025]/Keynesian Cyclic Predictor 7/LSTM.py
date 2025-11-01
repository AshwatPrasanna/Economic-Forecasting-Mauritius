import tensorflow as tf
import os
import pandas as pd
from numpy import *
import random

from Training_Data import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import load_model
import matplotlib.pyplot as plt


Mins = [1995.00, -14.50, 70246000000.00, 53980.22, 64841710004.00, 15559900000.00, -46324000000.00]
Maxes = [2023.00, 8.90, 663000000000.00, 531175.00, 615433000000.00, 202128000000.00, 81883000000.00]

def norm(x):
    for i in range(0, len(x)):
        minV = Mins[i]
        maxV = Maxes[i]
        x[i] = (((x[i]-minV)/(maxV-minV))*2.0)-1.0

    return x



def getArrays(timeseries, windowSize):
    X1 = []
    Y1 = []

    for i in range(0, len(timeseries)-windowSize):
        X1.append([[x] for x in timeseries[i:i+windowSize]])
        Y1.append(timeseries[i+windowSize])

    res = list(zip(X1, Y1))
    random.shuffle(res)
    X1, Y1 = zip(*res)

    return array(X1), array(Y1)



def trainModelAndStore(timeSeries, window, train_test_split=0.9, filePath="model"):
    X1, Y1 = getArrays(timeSeries, window)

    # Boilerplate source code: Greg Hogg
    

    split = int(train_test_split * len(timeSeries)) 


    X_train1, y_train1 = X1[:split], Y1[:split]
    X_val1, y_val1 = X1[split:], Y1[split:]


    ##### PLAY WITH MODEL PARAMETERS HERE ############

    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(8))
    model.add(Flatten())
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    
##    model = Sequential()
##    model.add(InputLayer((window, 1)))
##
##    model.add(LSTM(32, return_sequences=True))
##    model.add(LSTM(16))
##    
##    model.add(Dense(8, 'relu'))
##    model.add(Dense(5, 'sigmoid'))
##
##    model.add(Dense(1, 'linear'))

    ##################################################


    cp1 = ModelCheckpoint((filePath + '.keras'), save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.015), metrics=[RootMeanSquaredError()])
    model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=100, callbacks=[cp1])

    model.save((filePath + '.keras'))

    return model



def retrieveModel(filePath):
    model1 = load_model((filePath + '.keras'))
    return model1

def makePredictions(model, X):
    return float(model.predict(X).flatten())

#
#w = 15
#
#data = readData("Data.csv")
#GDP = array([norm(row)[2] for row in data])# array(list(range(0, 200)))# ##
#
#
#model = trainModelAndStore(GDP, w, filePath="modelTest")
#
#
#predictions = list(GDP[:w])
#
#for i in range(0, len(GDP)-w):
#    dataW = GDP[i:i+w]
#    predictions.append(makePredictions(model, array([[[x] for x in dataW]])))
#
#for i in range(0, 80):
#    dataW = predictions[-w:]
#    predictions.append(makePredictions(model, array([[[x] for x in dataW]])))
#
#y_pred = pd.Series(predictions)
#y_true = pd.Series(GDP)
#
#y_pred.plot()
#y_true.plot()
#
#plt.show()
