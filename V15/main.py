import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox, yeojohnson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from scipy.special import inv_boxcox
import os
import time
import numbers


from RFBNN_GDP import *

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.saving import *
import numpy as np

###### HYPER ############
maxOrder = 4
train_split = 0.8
start = 4
GRUWindow = 4
numColumns = 36 # Not including Year
d = 0.50
#########################

data = pd.read_csv('Data.csv')
mins = []
maxes = []

def resetData():
    global data
    global mins
    global maxes

    mins = []
    maxes = []

    data = pd.read_csv('Data.csv')

    for col in data.columns[1:]:
        mins.append(min(list(data[col])))
        maxes.append(max(list(data[col])))
        data[col] = (data[col] - min(list(data[col]))) / (max(list(data[col])) - min(list(data[col]))) + d

resetData()

idealOrders = []


def safeInvBoxcox(X, Lambda):
    v = [inv_boxcox(max(float(x), (-1.0/Lambda) + 0.2), Lambda) for x in X]
    return v



def safeBoxcox(X):
    for i in X:
        if i <= 0:
            print("FOUND NON-POSITIVE NUMBER!!!")
        if np.isnan(i):
            print("FOUND NAN!!!")
        if (not isinstance(i, numbers.Real)):
            print("FOUND COMPLEX NUMBER!!!")
            
    return boxcox([x+(0.01 - min(X)) for x in X])


#### Data has been collected, read, and normalised
def arima_model_for_row(i, end=None, return_order=False):
    name = 'P'+str(i)
    if name in data.columns:
        print(data[name])
        data[str(name)+'_boxcox'], Lambda = safeBoxcox(data[name])
    else:
        raise KeyError("Column", name, "is not found in the dataset.")

    #### This has taken care of the variance for us. Now, we find the difference we need

    idealOrder = None

    if return_order:
        print("Finding the ideal order for", name)
        model = pm.auto_arima(list(data[(name + '_boxcox')]), seasonal=False, start_p=0, start_q=0,
                              max_order=maxOrder, test='adf', error_action='ignore', suppress_warnings=True,
                              stepwise=True, trace=True)
        idealOrder = model.order
    else:
        idealOrder = idealOrders[i-1]

    #### Now we have the right order, so we construct the final ARIMA
    print("model trained on", data[(name+'_boxcox')][:end])
    model = ARIMA(data[(name+'_boxcox')][:end], order=idealOrder).fit()

    if return_order:
        return model, Lambda, idealOrder
    else:
        return model, Lambda


## Download the ideal orders here

if os.path.exists("idealOrders.txt"):
    idealOrders = [eval(string) for string in open("idealOrders.txt", "r").read().split("|")]
else:
    print("Ideal orders not found. Creating a file with the ideal orders for you. Please wait for a while.")
    idealOrders = []
    for i in range(1, len(data.columns)):
        model, Lambda, order = arima_model_for_row(i, return_order=True)
        idealOrders.append(order)
    f = open("idealOrders.txt", "w")
    f.write("|".join([str(x) for x in idealOrders]))
    f.close()
    print("Ideal orders file created.")


##### Some sample testing code

def get_first_residuals(j):
    residuals = [0.0] * start

    for i in range(start, len(list(data["Year"]))):
        model, Lambda = arima_model_for_row(j, end=i)
        boxcox_forecasts = model.forecast(1)
        print("Model predicted ", boxcox_forecasts)
        print("start safe inv bc GFR", j, "i is", i)
        forecast = list(safeInvBoxcox(boxcox_forecasts, Lambda))[0]
        print("end safe inv bc GFR")
        residuals.append(data["P"+str(j)][i] - forecast)

    return residuals


def forecast_residuals(j, residuals, length):
    model = None

    if not os.path.exists("GRUModel"+str(j)+".keras"):
        print("Training the GRU Model")

        model = Sequential([
            GRU(32, activation='relu', return_sequences=False, input_shape=(GRUWindow, 1)),
            #GRU(32, activation='relu'),
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
    return (x - d) * (maxes[i-1] - mins[i-1]) + mins[i-1]



def getRowForecast(j, length):
    initialResiduals = get_first_residuals(j)
    residuals = forecast_residuals(j, initialResiduals, length)

    model, Lambda = arima_model_for_row(j)
    boxcox_forecasts = model.forecast(length)
    forecasts = safeInvBoxcox(boxcox_forecasts, Lambda)
    predictions = np.add(np.array(forecasts), np.array(residuals[len(data["Year"]):]))

    return [denorm(x, j) for x in (list(data["P"+str(j)]) + list(predictions))]


def getForecast(length):
    transposedlist = []
    transposedlist.append(list(range(int(list(data["Year"])[0]), int(list(data["Year"])[-1] + length + 1))))
    for i in range(1, numColumns+1):
        transposedlist.append(getRowForecast(i, length))

    return list(np.array(transposedlist).T)

def getForecastWGDP(length, out_norm=False):
    model = retrieveModel(mins, maxes)
    
    transposedlist = []
    transposedlist.append(list(range(int(list(data["Year"])[0]), int(list(data["Year"])[-1] + length + 1))))
    transposedlist.append([])
    for i in range(2, numColumns+1):
        transposedlist.append(getRowForecast(i, length))

    # At this point, we have populated the entire chart in 1995, 1996, ..., 2023 \n V1 V2 V3 \n ...
    # Now, we put in the corrected GDP into the second element in the chart

    correctedGDPs = []

    for i in range(0, len(data["Year"]) + length):
        inp = [row[i] for row in transposedlist[2:]]
        rowNorm = [[x] for x in norm(([0.0] + inp), mins, maxes)[1:]]
        GDP_norm = np.array(np.array(model.predict(np.array([rowNorm]))).flat)[0]
        GDP = denormWI(GDP_norm, GDPIndex, mins, maxes)
        correctedGDPs.append(GDP)

    transposedlist[1] = list((np.array(getRowForecast(1, length)) + np.array(correctedGDPs))/2)

    if not out_norm:
        return list(np.array(transposedlist).T)
    else:
        return [[l[0]] + norm(list(l)[1:], mins, maxes) for l in list(np.array(transposedlist).T)]



def getForecastWithGrowth(length):
    l = getForecast(length)

    # Adding GDP Growth as the last parameter
    l[0] = np.append(l[0], np.array(0.0))
    for i in range(1, len(l)):
        l[i] = np.append(l[i], np.array(100.0 * (l[i][1] - l[i-1][1])/(l[i-1][1])))

    return l


def getFullForecastGivenGDP(row, length):
    global data
    resetData()

    for i in range(0, len(row)): # Important: don't update mins or maxes here...
        row[i] = (row[i] - mins[i]) / (maxes[i] - mins[i]) + d

    data.loc[len(data)] = [int(list(data["Year"])[-1]+1)] + row # ARIMA need not be re-ordered and GRU is same; ARIMA predictions are dynamic anyways so no other changes need to be made

    return getForecastWithGrowth(length)


def getFullForecast(row, length):

    model = retrieveModel(mins, maxes)

    print("Successfully retrieved the model")

    # First, we normalise the row to predict GDP

    rowNorm = [[x] for x in norm(([0.0] + row), mins, maxes)[1:]]

    print("Asked to predict on ", rowNorm)

    GDP_norm = np.array(np.array(model.predict(np.array([rowNorm]))).flat)[0]

    GDP = denormWI(GDP_norm, GDPIndex, mins, maxes)

    print("Model predicted ", GDP)

    return GDP#(getFullForecastGivenGDP([GDP] + row, length))




# The following code serves to pre-load some extra coordinates (input by the user) and incorporate it into forecasting. This new coordinate must be provided in the form of a row
# In order to use this functionality, first call updateData(), with the row that is given by the user. All datapoints not specified by the user should be set to -20.00 (these will be automatically determined)
# Then, call the function to forecast as normal
# When you want to change the user-set values, call resetData() and repeat the above process.

resetData()



def updateData(row): # The row begins with the year

    resetData()

    norm_row = [row[0]] + norm(row[1:], mins, maxes)

    for index, ele in enumerate(norm_row):
        if abs(row[index] + 20) <= 0.01:
            norm_row[index] = -20

    
    yr = row[0]
    last_year = int(max(data["Year"]))
    max_ind = max(data.index)

    len_forecast = yr - last_year

    forecast = getForecastWGDP(len_forecast, out_norm=True)[-len_forecast:]

    resetData()

    
    
    for i in range (1, int(yr-last_year)):
        data.loc[max_ind + i] = forecast[i-1]
        # subscript by i-1 to access the values to be appended


    for index, ele in enumerate(norm_row):
        if abs(ele + 20) <= 0.01:
            norm_row[index] = forecast[-1][index]

    data.loc[max_ind + len_forecast] = norm_row

    print("Finished cleaning up the data; ready to begin forecasting")



def makeForecastWithUserInput(row, years_to_forecast): # The row should contain the input data in order, with year being the first entry. All unmentioned datapoints should be set to -20 and will automatically be filled. Please note that the number of years forecasted is counted from the user's set datapoint
    updateData(row)
    res = getForecastWGDP(years_to_forecast)
    resetData()
    return res
    
    



t0 = time.time()
# q = [468153, 171111, 100464, -67919]

print(getForecastWGDP(5))

# new = float(getFullForecast(q, 1))
# old = list(pd.read_csv('Data.csv')["P1"])[-1]
# growth = 100.0*((new-old)/old)

# print("Growth was predicted as", str(growth) + str("%"))
#print("Predicted GDP growth for year", l[-2][0], "is", int(1000*l[-2][-1])/1000.0)

t1 = time.time()

print("Ready for console input...")

#print("Time taken:", t1-t0, "seconds")
# ++, ++, ++, +
