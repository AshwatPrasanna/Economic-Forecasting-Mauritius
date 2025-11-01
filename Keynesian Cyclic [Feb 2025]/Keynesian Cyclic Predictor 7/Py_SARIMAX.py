from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

import warnings 
#warnings.filterwarnings("ignore")



def getModel(timeSeries, order):
    model = ARIMA(timeSeries, order=order)
    model.initialize_approximate_diffuse()
    fitModel = model.fit()
    return fitModel

def testModel(train, test, order):
    predicted = []

    for i in range(0, len(test)):
        predicted.append(array(getModel(train + predicted, order).forecast().flat)[0])

    return mean_squared_error(predicted, test)
        
        

def findIdealParameters(train, test):
    bestOrder = (1, 1, 1)
    bestRMSE = -1


    for p in range(0, min([3, max([len(train)-3, 0])])):
        for d in range(0, min([3, max([len(train)-3, 0])])):
            for q in range(0, min([3, max([len(train)-3, 0])])):
                try:
                    err = testModel(train, test, (p, d, q))
                    
                    if err < bestRMSE or bestRMSE < 0.0:
                        bestOrder = (p, d, q)
                        bestRMSE = err
                except:
                    continue
    return bestOrder


def continueSeriesModel(series, train_test_split=0.6): ## This does not re-find the order at every step. The order is pre-trained on the initial data, not including the final datapoint
    train_len = int(train_test_split*len(series))
    train = series[:train_len]
    test = series[train_len:]
    order = findIdealParameters(train, test)

    return order

def continueSeries(order, series, n, train_test_split=0.6):
    predicted = []
    for i in range(0, n):
        predicted.append(array(getModel(series + predicted, order).forecast().flat)[0])
    return predicted


### Sources: https://www.geeksforgeeks.org/python-arima-model-for-time-series-forecasting/ and official documentation
