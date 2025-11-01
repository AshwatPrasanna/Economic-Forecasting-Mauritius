from __future__ import division
import random
import math
import numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


from RFBNN_core import *
from PSO_RFBNN import *
from Training_Data import *

import matplotlib.pyplot as plt


def randX0(particles, length):
    x0list = []
    for i in range(0, particles):
      x0 = []
      for _ in range(0, length):
        x0.append(random.random())
      x0list.append(x0)
    return x0list

def fitnessFunction(position, keys, values):
    xtrain, xtest, ytrain, ytest = train_test_split(keys, values, train_size=0.90)
    model = fitness_model(position, xtrain, ytrain)
    return calculate_fit(model, xtest, ytest)

def train(keys, values):
    model = PSOModel(randX0(numParticles, dims), bounds, fitnessFunction, keys, values, maxiter=numIter)
    model.begin()
    FinalModel = fitness_model(model.posbestgroup, keys, values)
    return FinalModel

models = []

def trainModel(keys, values):

    print("Model to predict parameter", i)

    res = train(keys, values)
    FinalModel = res[0]
    convergence = res[1]

    print("MAE:", MAE(FinalModel, keys, values))
    print("RMSE:", RMSE(FinalModel, keys, values))
    print("MAPE:", MAPE(FinalModel, keys, values))
    print("R2", R2(FinalModel, keys, values))


    l = []
    for arr in list(FinalModel.centres):
        l2 = (list(arr))
        l3 = []
        for ele in l2:
            l3.append(round(ele, 3))
        l.append(l3)


    ##
    print("Centers:", l)
    print("Weights:", list(FinalModel.weights))
    print("Betas:", FinalModel.inputBetas)
    print("Scaling:", FinalModel.scaling)
    
    model.append(FinalModel)

    print("--------------------------------------------------------\n\n\n")

data = readData("Data.csv")

def trainForParameter(i):
    keys = []
    values = []
    keys.append(data[0][:i] + data[0][(i+1):])
    for k in range(1, len(data)):
        values.append(data[k][i])
        keys.append(data[k][:i] + data[k][(i+1):])

    keys = keys[:-1]

    print(values)
    model = train(keys, values)
    return model

def GDPTrain():
    keys = [point[:-1] for point in data]
    values = [point[-1] for point in data]
    return train(keys, values)

FutureModels = []
MainModel = None

def trainCyclicModels():
    global FutureModels
    
    FutureModels = []

    for i in range(0, len(data[0])):
        FutureModels.append(trainForParameter(i))

    #MainModel = GDPTrain()

trainCyclicModels()


def getForecastOneYear(row): # Row is the input parameter
    forecast = []

    for i in range(0, len(data[0])):
        inputVal = row[:i] + row[(i+1):]
        forecast.append(numpy.array(numpy.array(FutureModels[i].predict(inputVal)).flat)[0])

    #trueForecastGDP = numpy.array(numpy.array(MainModel.predict(forecast[:-1])).flat)[0]

    #forecast[-1] = float(forecast[-1] + trueForecastGDP)/2.0

    return forecast


def getForecastNYears(n):
    # Where it should continue from - data from the textfields
    forecasted = []

    for _ in range(0, n):
        f = getForecastOneYear(data[-1])
        forecasted.append(f)
        data.append(f)
        trainCyclicModels()
        
    return forecasted

## Running getForecastNYears(n) returns a tuple consisting of all required datapoints in the order given in the excel sheet


getForecastNYears(2)




