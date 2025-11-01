import warnings
import matplotlib.pyplot as plt
import pandas as pd
from Hyperparams import *
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
from darts.models import TCNModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.missing_values import fill_missing_values
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

logging.disable(logging.CRITICAL)
print("\n\n\n\n\n")
from SheetPredictor import *
from RFBNN import *

########################################################################

# We must completely rescale the data since all the GDP measures may still be distinct from each other

RFBNNInputScaler = None
RFBNNOutputScaler = None # This is the scaler that will always be used when GDPs are involved in this file, since it is static and dependent only on the training data
DifferencedANNScaler = None # Doesn't include GDP or Growth as columns, sorted in lexicographical order
    
ANNforGDPGrowth = None
    
def trainRFBNN():
    global dataFrames
    global val_cutoff_index
    global RFBNNOutputScaler
    
    altogetherDF = None
    edited = False
    for index, df in enumerate(dataFrames):
        df2 = df.copy()
        df2 = df2.drop("RealGDPGrowth", axis="columns")
        df2 = df2.drop("GDP at current market prices", axis="columns")
        if index != 0:
            df2["Year"] = df2["Year"].astype('str')
            df2 = df2.drop("Year", axis="columns")
         
        if not edited:
            altogetherDF = df2
            edited = True
        else:
            altogetherDF = pd.concat([df2, altogetherDF], join="outer", axis=1)
            
         
        #modifiedDFs.append(df2)
    #altogetherDF = pd.concat(modifiedDFs, axis=1) # This is a combined, unscaled dataframe with all the required training info for the RFBNN
    print(altogetherDF.columns)
    #First, turn this into a timeseries, scale it, and store the scaler
    #print(altogetherDF.drop_duplicates(axis=1))
    fullTS = TimeSeries.from_dataframe(altogetherDF, time_col="Year", fill_missing_dates=True, freq=None) # For all available years
    
    RFBNNInputScaler = Scaler()
    fullTS_normalized = RFBNNInputScaler.fit_transform(fullTS)
    
    # Then, determine the correct values of the GDPs, scale it, and store the scaler
    GDPList = np.array(dataFrames[0]["GDP at current market prices"], dtype=int) # For all available years
    RFBNNOutputScaler = StandardScaler()
    #GDPList.transpose()
    GDPList = GDPList.reshape(-1, 1)
    fullGDP_normalized = RFBNNOutputScaler.fit_transform(GDPList)
    
    # Train, validation split - this time, it's random
    x_train, x_test, y_train, y_test = train_test_split(fullTS_normalized.all_values(), fullGDP_normalized, test_size=0.20)
    
    # Then, train the RFBNN
    print("\n\n\n")
    print(np.array([arr.flat for arr in list(fullTS_normalized.all_values())]))
    print("\n\n\n")
    print(fullGDP_normalized.flatten())
    print("\n\n\n")
    #model = train_rbfnn(fullGDP_normalized, fullTS_normalized.all_values())
    
    return 0#model
    

def getGDPPredictionFrameForValidationData(): # This is for the ANN
    # Get the GDP predictions from each of the different sheets, all in a dedicated dataframe
    GDP_predictor_scores = [] # This is a 2-dimensional array, the inner dimension gives horizontal rows, all the GDP predictor variables, and the vertical one is through time. It will be backwards first, then transposed.
#    print("zzz")
#    print(forecastValidationDataFromTrainingDataOnly(0))
#    print(forecastValidationDataFromTrainingDataOnly(1))
#    print("zzz")

    for i in range(0, len(sheetNames)):
        df, unscaledTS = forecastValidationDataFromTrainingDataOnly(i) # We care about the unscaled TS
        GDP_predictor_scores.append(list((unscaledTS.to_dataframe())["GDP at current market prices"]))
    GDP_predictor_scores = np.array(GDP_predictor_scores)
    GDP_predictor_scores = GDP_predictor_scores.transpose() # From validation set only
    
    print("Sampling GDP predictor scores (X) \n -------------------------------")
    print(GDP_predictor_scores)
    print("---------------------------\n\n")
    
    # Now, GDP_predictor_scores gives us the x-data, increasing with time, all of which will go to training (no validation for this ANN)
    
    GDPValues = list(dataFrames[0]["GDP at current market prices"])[val_cutoff_index:] # These are the true values, the Y set
    
    GDPValues = np.array(GDPValues).reshape(-1, 1)
    
    print("Real GDP Values (Y)\n -------------------------------")
    print(GDPValues)
    print("---------------------------\n\n")
    
    scaledInputs = np.array([RFBNNOutputScaler.transform(np.array(row).reshape(-1, 1)) for row in GDP_predictor_scores.tolist()])
    
    print("Scaling Data Uniformly\n -------------------------------")
    print(scaledInputs)
    print("---------------------------\n\n")
    
    scaledOutputs = RFBNNOutputScaler.transform(np.array(GDPValues).reshape(-1, 1))
    
    print("Scaling Outputs Uniformly\n -------------------------------")
    print(scaledOutputs)
    print("---------------------------\n\n")
    
    #GDP_predictor_scores = np.array(GDP_predictor_scores).reshape(-1, 1)
    
    #print(GDP_predictor_scores, "Output from the event")
    
    x_train, x_val, y_train, y_val = train_test_split(scaledInputs, scaledOutputs, test_size=0.15)
    
    return x_train, x_val, y_train, y_val # These are normalized using the default scaler for this Python file, since all values are indeed GDP
    
    # Define a dedicated scaler for this data, but fit it only to the given data, not the forecasted data. The scaler's bounds should never change
    # Output the GDP prediction frame ready for ANN use, and the scaler

def trainANNforGDP():

    x_train, x_val, y_train, y_val = getGDPPredictionFrameForValidationData()
    
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)
    
    model = Sequential([
        Dense(len(x_train[0]), activation='linear', input_shape=(len(x_train[0]),)),  # Hidden layer
        Dense(2, activation='relu'),
        Dense(1, activation='linear')  # Output layer
    ])
    
    model.compile(optimizer='adam',
              loss='mape',
              metrics=['accuracy'])
              
    print("Xtrain")
    print(x_train)
    print("Ytrain")
    print(y_train)

    history = model.fit(x_train, y_train, epochs=300, callbacks=[callback], validation_data=(x_val, y_val))
    
    return model, history


def TrainANNForGDPGrowth(): # This takes the difference of TimeSeries

    global sheetNames
    global GDPGrowthScaler
    global DifferencedANNScaler
    global ANNforGDPGrowth

    UnscaledDFs = [] # Unscaled is used since there is a different scaler used in this Python file
    for name in sheetNames:
        unscaledDF = pd.read_csv(name+".csv") # Includes both training and validation data
        unscaledDF["Year"] = pd.to_datetime([str(int(year)) for year in unscaledDF["Year"]])
        unscaledDF.set_index("Year")
        unscaledDF = unscaledDF.drop("RealGDPGrowth", axis="columns")
        unscaledDF = unscaledDF.drop("GDP at current market prices", axis="columns")
        UnscaledDFs.append(unscaledDF)
    
    fullUnscaledDF = pd.concat(UnscaledDFs, axis=1)
    fullUnscaledDF = fullUnscaledDF.drop("Year", axis="columns")
    fullUnscaledDF = fullUnscaledDF.diff()
    fullUnscaledDF = fullUnscaledDF.reindex(sorted(fullUnscaledDF.columns), axis=1)
    
    GDPGrowthScaler = StandardScaler()
    DifferencedANNScaler = StandardScaler()
    
    allDataX = DifferencedANNScaler.fit_transform(fullUnscaledDF.values)[1:]
    allDataY = GDPGrowthScaler.fit_transform(np.array(pd.read_csv(sheetNames[0]+".csv")["RealGDPGrowth"]).reshape(-1, 1))[1:]
    
    trainX, valX, trainY, valY = train_test_split(allDataX, allDataY, test_size=0.25)
    
    
    print("Predictors for GDP Growth\n -----------------------------------------------")
    print(allDataX)
    print("-----------------------------------------------")
    print(allDataY)
    print("-----------------------------------------------\n\n")
    

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)
    
    model = Sequential([
        Dense(len(trainX[0]), activation='linear', input_shape=(len(trainX[0]),)),  # Hidden layer
        Dense(10, activation='relu'),
        Dense(1, activation='linear')  # Output layer
    ])
    
    model.compile(optimizer='adam',
              loss='mape',
              metrics=['accuracy'])

    history = model.fit(trainX, trainY, epochs=300, callbacks=[callback], validation_data=(valX, valY))
    
    ANNforGDPGrowth = model
    
    return model, history



def getGDPCorrectedForecasts(N):
    global sheetNames

    model, _ = trainANNforGDP()
    #rbfnn = trainRFBNN()
    UnscaledDFs = [] # Unscaled is used since there is a different scaler used in this Python file
    GDPPredictors = []
    
    for index in range(0, len(sheetNames)):
        _, unscaled_ts = forecastSeries(index, N)
        unscaledDF = unscaled_ts.to_dataframe()
        GDPPredictors.append(unscaledDF["GDP at current market prices"])
        unscaledDF = unscaledDF.drop("RealGDPGrowth", axis="columns")
        unscaledDF = unscaledDF.drop("GDP at current market prices", axis="columns")
        UnscaledDFs.append(unscaledDF)
        
    fullUnscaledDF = pd.concat(UnscaledDFs, axis=1)
    GDPPredictors = np.array(GDPPredictors)
    GDPPredictors = GDPPredictors.transpose()
    
    #OLD: RFBNNOutputScaler.transform(np.array(GDPPredictors).reshape(-1, 1))#RFBNNOutputScaler.transform(np.array(GDPPredictors).reshape(-1, 1)) # These need to be scaled for ANN processing
    
    print("Unscaled GDP Psedictors\n -----------------------------------------------")
    print(GDPPredictors)
    print("-----------------------------------------------\n\n")
    
    GDPPredictors_scaled = [RFBNNOutputScaler.transform(np.array(row).reshape(-1, 1)) for row in GDPPredictors.tolist()]
    
    print("Input to the Model\n -----------------------------------------------")
    print(GDPPredictors_scaled) # This is scaled
    print("-----------------------------------------------\n\n")
    print("Output from the Model\n -----------------------------------------------")
    scaledOutputList = [model.predict(np.array(row).reshape(-1,len(sheetNames))).flatten()[0] for row in GDPPredictors_scaled]
    print(scaledOutputList) # This is scaled
    print("-----------------------------------------------\n\n")
    
    ANN_predictions = RFBNNOutputScaler.inverse_transform(np.array(scaledOutputList).reshape(-1, 1)).flatten() # Gives a list of GDP values
    average_predictions = RFBNNOutputScaler.inverse_transform(np.array([np.mean(row) for row in GDPPredictors_scaled]).reshape(-1, 1)).flatten()
    
    #ANN_predictions = np.array(ANN_predictions) # These are unscaled, like the rest of the data
    
    print("Unscaled Output from the Model\n -----------------------------------------------")
    print(ANN_predictions)
    print("-----------------------------------------------\n\n")
    
    print("Unscaled average\n -----------------------------------------------")
    print(average_predictions)
    print("-----------------------------------------------\n\n")
    
    #RFBNNInput = RFBNNInputScaler.transform(fullUnscaledDF).values # Takes all the non-replaced forecast values, scales them for RFBNN input, stored as a numpy matrix
    #RFBNNPredictions_unscaled = np.array(rbfnn.predict(list(RFBNNInput)))
    
    #RFBNNPredictions = np.array(RFBNNInputScaler.inverse_transform(RFBNNPredictions_unscaled)) # Unscaled, like everything else, multiple values
    
    GDPColumnUnscaled = (ann_weightage*ANN_predictions) + ((1-ann_weightage)*average_predictions) #(ann_weightage*ANN_predictions) #+ ((1-ann_weightage)*0)#RFBNNPredictions) # These are the best GDP prediction values for future years
    GDPColumnUnscaled2 = (0.2*ANN_predictions) + (0.8*average_predictions)
    GDPColumnUnscaled3 = (0.8*ANN_predictions) + (0.2*average_predictions)
    
    print("0.20 ANN Weightage\n -----------------------------------------------")
    print(GDPColumnUnscaled2)
    print("-----------------------------------------------\n\n")
    
    print("0.80 ANN Weightage\n -----------------------------------------------")
    print(GDPColumnUnscaled3)
    print("-----------------------------------------------\n\n")
    
    
    
    # Now, add the GDP Growth, calculated as the Market Price GDP rate of growth minus the GDP deflator
    #print(fullUnscaledDF.columns)
    
    
    ######################## Correct the forecast with the ANN prediction for GDP Growth ##############################
    
#    GDPDeflators = list(fullUnscaledDF["GDP deflator"])
#    GDPColumnIncludingLastVal = [list(dataFrames[0]["GDP at current market prices"])[-1]] + list(GDPColumnUnscaled)
#    RealGrowthRate = [] # Predicted indirectly using AI
#    
#    print(len(GDPColumnUnscaled), len(GDPColumnIncludingLastVal))
#    
#    for i in range(0, len(GDPColumnUnscaled)):
#        growthRateNominal = 100.0*(GDPColumnIncludingLastVal[i+1]-GDPColumnIncludingLastVal[i])/(GDPColumnIncludingLastVal[i])
#        deflator = GDPDeflators[i]
#        RealGrowthRate.append(growthRateNominal - deflator)
#        
#    
#    fullUnscaledDF.insert(0, "RealGDPGrowth", RealGrowthRate, True)
#    return fullUnscaledDF



    # First, get the last row, then concatenate all the predicted values. All unscaled
    
    GCUnscaledDFs = [] # Unscaled is used since there is a different scaler used in this Python file
    for name in sheetNames:
        unscaledDF = pd.read_csv(name+".csv") # Includes both training and validation data
        unscaledDF["Year"] = pd.to_datetime([str(int(year)) for year in unscaledDF["Year"]])
        unscaledDF.set_index("Year")
        unscaledDF = unscaledDF.drop("RealGDPGrowth", axis="columns")
        unscaledDF = unscaledDF.drop("GDP at current market prices", axis="columns")
        GCUnscaledDFs.append(unscaledDF)
    
    GCfullUnscaledDF = pd.concat(UnscaledDFs, axis=1)
    #GCfullUnscaledDF = GCfullUnscaledDF.drop("Year", axis="columns")
    GCfullUnscaledDF = GCfullUnscaledDF.diff()
    GCfullUnscaledDF = GCfullUnscaledDF.reindex(sorted(GCfullUnscaledDF.columns), axis=1)
    
    lastRow = GCfullUnscaledDF.iloc[-1]
    
    # Then take the differences
    
    #fullUnscaledDF2 = fullUnscaledDF.set_index("Year")
    #fullUnscaledDF2.drop("Year", axis="columns")
    fullUnscaledDF2 = fullUnscaledDF.reindex(sorted(fullUnscaledDF.columns), axis=1)
    fullUnscaledDF2 = pd.concat([lastRow, fullUnscaledDF2]).iloc[:, 1:]

    vals = np.array([(lastRow.values)] + list(fullUnscaledDF2.values))
    vals = vals[~np.isnan(vals).any(axis=1), :]
    
    diffs = np.diff(vals, axis=0)
    
    # Then scale
    
    GrowthCalcInput = DifferencedANNScaler.transform(diffs) # in correct order, not including year, gdp, or growth, as needed, from 2020 onwards
    UnscaledPredictions = [GDPGrowthScaler.inverse_transform(ANNforGDPGrowth.predict(np.array(row).reshape(-1, len(row))).reshape(-1, 1)).flatten()[0] for row in list(GrowthCalcInput)]
    
    print(UnscaledPredictions)
    
    # Add them back to the correct years
    
    fullUnscaledDF.insert(0, "RealGDPGrowth", UnscaledPredictions, True)
    
    fullUnscaledDF.insert(0, "GDP at current market prices", GDPColumnUnscaled, True) # This is unscaled, just as it should be for the unscaled dataframe
    
    return fullUnscaledDF

def test(forecast): # Forecast should be a full unscaled DF
    df = pd.read_csv("data_testing.csv")
    lastYear = int(list(pd.read_csv(sheetNames[0]+".csv")["Year"])[-1])
    forecast = forecast.iloc[(2023-(lastYear+1)):(2025-(lastYear+1))]
    
    df = df.dropna(axis=1, how="any")
    df = df.drop("Year", axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
        
    forecast = forecast.drop(forecast.columns.difference(df.columns), axis=1)
    forecast = forecast.reindex(sorted(forecast.columns), axis=1)
    # They should have the same columns now
    
    print("True Data\n -----------------------------------------------")
    print(df.columns)
    print("Predicted Data\n -----------------------------------------------")
    print(forecast.columns)
    print("-----------------------------------------------\n\n")
    
    d = df.to_numpy().reshape(2, -1)
    p = forecast.to_numpy().reshape(2, -1)
    
    du23 = d.tolist()[0]
    du24 = d.tolist()[1]
    pu23 = p.tolist()[0]
    pu24 = p.tolist()[1]
    
    
    scl = StandardScaler()
    dS = scl.fit_transform(d)
    pS = scl.transform(p)
    
    print("True Data\n -----------------------------------------------")
    print(d)
    print("Predicted Data\n -----------------------------------------------")
    print(p)
    print("-----------------------------------------------\n\n")
    
    d23 = dS.tolist()[0]
    d24 = dS.tolist()[1]
    p23 = pS.tolist()[0]
    p24 = pS.tolist()[1]
    
    print("R^2 for 2023 is:", r2_score(du23, pu23))
    print("R^2 for 2024 is:", r2_score(du24, pu24))
    print("MAPE for 2023 is:", str(mean_absolute_percentage_error(du23, pu23)) + "%")
    print("MAPE for 2024 is:", str(mean_absolute_percentage_error(du24, pu24)) + "%")
    
    
    
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    
    

########################################################################
#
#def getDifferencedData():
#    # Take the predicted, GDP-corrected forecasts and take their difference
#    # Add GDP Deflator forecast, non-differenced as a column to the data
#    
#def trainANNforGrowth():
#    # Train from the above differenced data
#
#def getFullForecast():
#    # Combine two above functions, as well as the simple subtraction calculation with a weightage factor
#    

########################################################################
trainRFBNN()
TrainANNForGDPGrowth()
#trainANNforGDP()
fc = getGDPCorrectedForecasts(10)
fc.to_csv('ForecastV1.4.csv', index=False)
test(fc)
