from numpy import *
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from ForestARIMA import *
from Training_Data import *
import math

GDPSeries = read_csv("Data.csv", date_format="%Y", parse_dates=[0], index_col = 0)
GDPvalues = readData("Data.csv")[1]
GDPparameters = readData("Data.csv")[0]

results = []
ideal = [0.7] * 10

for i in range(0, 10):

    regressors = list(orderRegressors(GDPvalues, GDPparameters))

    print("Determined the Random Forest Regressors")

    results.append(PredictWithRegressors(GDPvalues, GDPparameters[-1], regressors))

print(sum(results)/len(results))
