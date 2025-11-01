from __future__ import division
import random
import math
import numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd


from RFBNN_core import *
from PSO_RFBNN import *
from Training_Data import *
from Py_SARIMAX import *

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

Mins = [1995.0, -14.5, 70246000000.0, 475000.0, 16108000000.0, 41205000000.0, 42908000000.0, 8134000000.0, 24300.0, 475000.0, 8200.0, 0.5, 359300000.0, 179780.0, 7472000000.0, 15559900000.0, 21605000000.0, 129316.0, 5888360383.56, 19815172646.61, 37910198671.2, 0.5, 13291438000.0, -85821700000.0, 16499128200.0, 0.38, 0.65, 17.8, 0.03, 2.64, 303.39, 825276530.0, 292124.04]
Maxes = [2023.0, 8.9, 663000000000.0, 592800.0, 118000000000.0, 347837000000.0, 360000000000.0, 826000000000.0, 52200.0, 592800.0, 31800.0, 10.8, 36603000000.0, 1399408.0, 85993000000.0, 202128000000.0, 516500000000.0, 331924.0, 27524000000.0, 121069000000.0, 430237000000.0, 10.8, 43901532220.53, -1623100000.0, 94655654800.0, 1.23, 1.18, 44.98, 0.03, 9.27, 3342.39, 4287566107.1, 1291075964.38]
ARIMA_coefficient = 0.5

#### Trained models are stored here in their exported formats

FutureModelsEx = []
MainModelEx = None
GrowthModelEx = None
ARIMAModels = []

####################################### BELOW not to be modified

FutureModels = [importRFBN(x) for x in FutureModelsEx]
if MainModelEx != None and GrowthModelEx != None:
    MainModel = importRFBN(MainModelEx)
    GrowthModel = importRFBN(GrowthModelEx)
else:
    MainModel = None
    GrowthModel = None

#######################################

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
fullData = readData("DataFull.csv")
NaNValue = -20.0
delta = 0.01

def trainForParameter(i):
    keys = []
    values = []
    keys.append(data[0][:i] + data[0][(i+1):])
    for k in range(1, len(data)):
        values.append(data[k][i])
        keys.append(data[k][:i] + data[k][(i+1):])

    keys = keys[:-1]

    model = train(keys, values)
    return model

def GDPTrain():
    keys = [point[0:1] + point[2:] for point in data]
    values = [point[1] for point in data]
    return train(keys, values)

def GrowthTrain():
    keys = [point[:2] + point[3:] for point in data]
    values = [point[2] for point in data]
    return train(keys, values)


def trainCyclicModels(t_t_split=0.60):
    global FutureModels
    global MainModel
    global GrowthModel
    global ARIMAModels
    
    FutureModels = []
    ARIMAModels = []

    for i in range(0, len(data[0])):
        FutureModels.append(trainForParameter(i))
        dataSpecific = [row[i] for row in data]
        ARIMAModels.append(continueSeriesModel(dataSpecific))
        print("\n\n----------------------\n\nMARK: Finished training parameter", i+1, "\n\n----------------------\n\n")

    MainModel = GDPTrain()
    GrowthModel = GrowthTrain()

    


def getForecastOneYear(row): # This doesn't use ARIMA
    forecast = []

    for i in range(0, len(data[0])):
        inputVal = row[:i] + row[(i+1):]
        forecast.append(numpy.array(numpy.array(FutureModels[i].predict(inputVal)).flat)[0]) # time-step parameter prediction

    forecast[0] = row[0] + (data[1][0] - data[0][0]) # Update the year reading correctly

    trueForecastGDP = numpy.array(numpy.array(MainModel.predict(forecast[0:1] + forecast[2:])).flat)[0] # predict true GDP
    trueForecastGrowth = numpy.array(numpy.array(GrowthModel.predict(forecast[:2] + forecast[3:])).flat)[0] # predict true growth

    forecast[-1] = trueForecastGDP # set true GDP

    return forecast


def replaceFullData():
    global data
    global NaNValue
    global fullData
    
    mainD = data[:]
    forecasted = [] # This is the replaced version

    for row in range(0, len(fullData)):
        f = getForecastOneYear(mainD[-1])
        # Fix the version here by combining
        f = [(point if (abs(fullData[row][index] - NaNValue) <= delta) else fullData[row][index]) for index, point in enumerate(f)]
        
        forecasted.append(f)
        mainD.append(f)
        #trainCyclicModels()
        
    fullData = forecasted
    data = data + fullData



def getForecastNYears(mainD, n):
    # Where it should continue from - data from the textfields
    forecasted = []

    for _ in range(0, n):
        f = getForecastOneYear(mainD[-1])
        forecasted.append(f)
        mainD.append(f)
        #trainCyclicModels()
        
    ## Now, forecast using ARIMA
    rawARIMA = []

    for indx in range(0, len(mainD[-1])):
        rawARIMA.append(continueSeries(ARIMAModels[indx], [row[indx] for row in mainD], n))

    ARIMA_pred = numpy.array(rawARIMA).T # rows are the inner dimension

    return list(numpy.add(ARIMA_pred*ARIMA_coefficient, numpy.array(forecasted)*(1-ARIMA_coefficient)))

## Running getForecastNYears(n) returns a tuple consisting of all required datapoints in the order given in the excel sheet



def predictForPast(plot=False, extra=10, ind=2):
    predicted = []
    true = []
    
    for k in range (5, len(data)-1): # K is the index to start from
        predicted.append(getForecastNYears(data[0:(k+1)], 1)[0][ind]) # k+1 elements before
        true.append(data[k+1][ind])

    extra_pred = getForecastNYears(data[:], extra)
    predicted += [l[ind] for l in extra_pred]

    if plot:
        y_pred = pd.Series(predicted)
        y_true = pd.Series(true)
        y_pred.plot()
        y_true.plot()
        plt.show()
        

    return predicted, true



def denorm(x):
    for i in range(0, len(x)):
        minV = Mins[i]
        maxV = Maxes[i]
        x[i] = (((x[i]+1.0)/2.0) * (maxV - minV)) + minV

    return x



    


####################################


