import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from scipy.special import inv_boxcox
import os

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.saving import *
import numpy as np

###### HYPER ############
maxOrder = 5
train_split = 0.8
start = 10
GRUWindow = 10
#########################

data = pd.read_csv('Data.csv')
mins = []
maxes = []

for col in data.columns[1:]:
    mins.append(min(list(data[col])))
    maxes.append(max(list(data[col])))
    data[col] = (data[col] - min(list(data[col]))) / (max(list(data[col])) - min(list(data[col]))) + 0.01

#### Data has been collected, read, and normalised
def arima_model_for_row(i, end=None):
    name = 'P'+str(i)
    if name in data.columns:
        data[str(name)+'_boxcox'], Lambda = boxcox(data[name])
    else:
        raise KeyError("Column", name, "is not found in the dataset.")

    #### This has taken care of the variance for us. Now, we find the difference we need

    model = pm.auto_arima(list(data[(name + '_boxcox')]), seasonal=False, start_p=0, start_q=0,
                              max_order=maxOrder, test='adf', error_action='ignore', suppress_warnings=True,
                              stepwise=True, trace=True)

    idealOrder = model.order

    #### Now we have the right order, so we construct the final ARIMA
    model = ARIMA(data[(name+'_boxcox')][:end], order=idealOrder).fit()

    return model, Lambda



##### Some sample testing code

def get_first_residuals(j):
    residuals = [0.0] * start

    for i in range(start, len(list(data["Year"]))):
        model, Lambda = arima_model_for_row(j, end=i)
        boxcox_forecasts = model.forecast(1)
        forecast = list(inv_boxcox(boxcox_forecasts, Lambda))[0]
        residuals.append(data["P"+str(j)][i] - forecast)

    return residuals


def forecast_residuals(j, residuals, length):
    model = None

    if not os.path.exists("GRUModel"+str(j)+".keras"):
        model = Sequential([
            GRU(64, activation='relu', return_sequences=True, input_shape=(GRUWindow, 1)),
            GRU(32, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse')

        X = []
        Y = []

        for indx in range(0, len(residuals) - GRUWindow):
            X.append(residuals[indx:indx+GRUWindow])
            Y.append(residuals[indx+GRUWindow])

        X = np.array(X)
        Y = np.array(Y)

        model.fit(X, Y, epochs=100, batch_size=10)
        model.save("GRUModel"+str(j)+".keras")

    else:
        model = load_model("GRUModel"+str(j)+".keras")


    for i in range(0, length):
        residuals.append(float(model.predict(np.array([residuals[-GRUWindow:]]))))

    return residuals

def denorm(x, i):
    return (x - 0.01) * (maxes[i-1] - mins[i-1]) + mins[i-1]



def getRowForecast(j, length):
    initialResiduals = get_first_residuals(j)
    residuals = forecast_residuals(j, initialResiduals, length)

    model, Lambda = arima_model_for_row(j)
    boxcox_forecasts = model.forecast(length)
    forecasts = inv_boxcox(boxcox_forecasts, Lambda)
    predictions = np.add(np.array(forecasts), np.array(residuals[len(data["Year"]):]))

    return [denorm(x, j) for x in (list(data["P"+str(j)]) + list(predictions))]


def getForecast(length):
    transposedlist = []
    transposedlist.append(list(range(list(data["Year"])[0], list(data["Year"])[-1] + length + 1)))
    for i in range(1, len(data.columns)):
        transposedlist.append(getRowForecast(i, length))

    return list(np.array(transposedlist).T)

print(getForecast(30))
# In order to make a prediction, update the Data.csv file accordingly and call getForecast to get output in the same format
