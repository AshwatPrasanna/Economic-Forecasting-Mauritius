from numpy import *
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math


TimeSeries = read_csv("Data.csv", date_format="%Y", parse_dates=[0], index_col = 0)
Values = TimeSeries.values


def RollingForecast(train, test, order):
    history = [x for x in train] # Contains only the values, not the frequency
    predictions = []
    
    for t in range(0, len(test)):
        model = ARIMA(history, order=order)
        fit = model.fit()
        out = fit.forecast()
        predictions.append(out[0])
        history.append(out)

    return predictions
##
##trainFrac = 0.8
##size = int(trainFrac * len(Values))
##
##train = Values[:size]
##test = Values[size:]
##
##pred = RollingForecast(train, test, (4, 2, 1))
##print("RMSE:", math.sqrt(mean_squared_error(test, pred)))
        

def PredictWithRegressors(history, dataPoint, regressors):
    arr = array([array(dataPoint).flat])
    p = int(regressors[0].predict(arr))
    d = int(regressors[1].predict(arr))
    q = int(regressors[2].predict(arr))
    order = (p, d, q)
    print(order)

    model = ARIMA(history, order=order)
    fit = model.fit()
    return fit.forecast()[0]








### Sources: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/, https://www.datacamp.com/tutorial/arima, https://www.geeksforgeeks.org/python-datetime-strptime-function/, and official library documentation
