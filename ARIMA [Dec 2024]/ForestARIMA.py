### This version uses the following as Forest inputs: length, correlation hypothesis, and individual datapoints
from numpy import *
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from StandardARIMA import *
from RandomForest import *
from Training_Data import *
import math





#########################################


##def dataParameters(y, trainSize): #This will be used as an x-coordinate for the random forest regressor
##    lngth = min(int(len(y)*trainSize+10), len(y))
##    print(lngth)
##    train = y[:lngth]
##    test = y[lngth:]
##    p = adfuller(train)[1]
##
##    idealOrder = findIdealOrder(train, test, [[1, 5], [1, 4], [0, 3]]) # bounds are p, d, q
##
##    return ([lngth, p], idealOrder) # Need to add the initial list here
##    
    
def evaluateARIMA(train, test, order):
    pred = RollingForecast(train, test, order)
    return math.sqrt(mean_squared_error(test, pred))


def findIdealOrder(train, test, bounds): # bounds are [p], [d], [q]
    minErr = -1
    minOrder = None
    for p in range(bounds[0][0], bounds[0][1]):
        for d in range(bounds[1][0], bounds[1][1]):
            for q in range(bounds[2][0], bounds[2][1]):
                try:
                    err = evaluateARIMA(train, test, (p, d, q))
                except:
                    err = -1
                if (err < minErr or minErr == -1) and err != -1:
                    minErr = err
                    minOrder = (p, d, q)

    print("minErr:", minErr)
    return minOrder





###########################################

    


def orderRegressors(values, supportingParameters):

    dataPoints = []
    correctValuesP = []
    correctValuesD = []
    correctValuesQ = []



    for idx in range(10, len(supportingParameters)-1):
        inputVal = supportingParameters[idx] # input to the random forest
        
        targetVal = values[idx+1] # what the ARIMA should be getting

        ARIMAIn = values[:idx+1] # train for the ARIMA

        # Now, find the ideal order and set that as the desired value. Train with random forest for P, D, Q

        order = list(findIdealOrder(ARIMAIn, [targetVal], [[3, 5], [1, 4], [0, 3]]))

        dataPoints.append(inputVal)
        correctValuesP.append(order[0])
        correctValuesD.append(order[1])
        correctValuesQ.append(order[2])

        
    pReg = forestRegTrain(dataPoints, correctValuesP)
    dReg = forestRegTrain(dataPoints, correctValuesD)
    qReg = forestRegTrain(dataPoints, correctValuesQ)

    return (pReg, dReg, qReg)









    

# https://machinelearningmastery.com/time-series-data-stationary-python/
