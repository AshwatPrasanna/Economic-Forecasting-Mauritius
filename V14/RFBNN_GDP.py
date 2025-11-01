from __future__ import division
import random
import math
from numpy import *
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os

from RFBNN_core import *
from PSO_RFBNN import *
from Training_Data import *

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.saving import *
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt


def denorm(x, Mins, Maxes):
    for i in range(0, len(x)-1):
        minV = Mins[i]
        maxV = Maxes[i]
        x[i] = (x[i] * (maxV - minV)) + minV

    return x

def denormWI(x, i, Mins, Maxes):
    minV = Mins[i]
    maxV = Maxes[i]
    x = (x * (maxV - minV)) + minV

    return x


def norm(x, Mins, Maxes):
    for i in range(0, len(x)):
        minV = Mins[i]
        maxV = Maxes[i]
        x[i] = (x[i] - minV) / (maxV - minV)

    return x


GDPIndex = 0 ## GDP must always be the first value in the row


def randX0(particles, length):
    x0list = []
    for i in range(0, particles):
        x0 = []
        for _ in range(0, length):
            x0.append(random.random())
        x0list.append(x0)
    return x0list



def fitnessFunction(position, keys, values):
    xtrain, xtest, ytrain, ytest = train_test_split(keys, values, train_size=0.80)
    model = fitness_model(position, xtrain, ytrain)
    return calculate_fit(model, xtest, ytest)

def train(keys, values):
    model = PSOModel(randX0(numParticles, dims), bounds, fitnessFunction, keys, values, maxiter=numIter)
    model.begin()
    FinalModel = fitness_model(model.posbestgroup, keys, values)
    return FinalModel



localData = [row[1:] for row in readData("Data.csv")]

def normalise(mins, maxes):
    global localData
    localData = [norm(row, mins, maxes) for row in localData]

def GDPTrain():
    keys = [point[1:] for point in localData] #+ [point[1:] for point in localData][26:]
    values = [point[0] for point in localData] #+ [point[0] for point in localData][26:]

    #for ele in zip(keys, values):
    #   print(ele)

    return train(keys, values)


def GDPData():
    keys = [point[1:] for point in localData] #+ [point[1:] for point in localData][26:]
    values = [point[0] for point in localData] #+ [point[0] for point in localData][26:]

    return keys, values

def retrieveModel(mins, maxes):
    normalise(mins, maxes)
    # if not os.path.exists("GDPModel.txt"):
    #     model = GDPTrain()
    #     text = str(model.export())
    #     f = open("GDPModel.txt", "w")
    #     f.write(text)
    #     f.close()
    # f = open("GDPModel.txt", "r")
    # t = eval(f.read())
    # f.close()
    # return importRFBN(t)

    if not os.path.exists("GDPModel.keras"):
        model = Sequential()
        model.add(InputLayer((36,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(4, activation='leaky_relu'))
        model.add(Dense(4, activation='leaky_relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        K, V = GDPData()
        K = [[[x] for x in R] for R in K]
        V = [[x] for x in V]

        for i in range(0, len(K)):
            print(K[i], V[i])

        print("BP1")

        K = array(K)
        V = array(V)

        a, a_v, b, b_v = train_test_split(K, V, test_size=0.4)

        history = model.fit(a, b, epochs=500, batch_size=50, validation_data=(a_v, b_v))#, callbacks=[early_stopping])
        model.save("GDPModel.keras")

    model = load_model("GDPModel.keras")
    return model




