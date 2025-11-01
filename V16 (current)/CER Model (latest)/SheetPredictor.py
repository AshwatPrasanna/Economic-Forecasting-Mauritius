import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Hyperparams import *
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.missing_values import fill_missing_values
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from statsmodels.tsa.api import VAR
warnings.filterwarnings("ignore")
from pmdarima.arima import auto_arima, ADFTest
from prophet import Prophet
from sklearn.metrics import r2_score
import math

import logging
logging.disable(logging.CRITICAL)


class CERSubModel:
    def __init__(self, modelType, data):
        self.modelType = modelType
        self.data = data


def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }
    
def generateForecastingModels(): # For all sheets. Calling this causes all the models to retrain and refresh.
    global dataFrames
    global timeSeries
    global sheetNames
    global val_cutoff_index
    global input_chunk_len
    global output_chunk_len
    global TCNEpochs
    global dropout
    global dilation_base
    global kernel_size
    global num_filters
    
    # Preparing the scaled dataframes and time series
    
    scalers = []
    scaledDF = []
    scaledTS = []
    for index in range(0, len(sheetNames)):
        scaler = Scaler()
        sts = scaler.fit_transform(timeSeries[index])
        scaledTS.append(sts)
        scalers.append(scaler)
        sdf = sts.to_dataframe()
        scaledDF.append(sdf)
        
        
    # Preparing the TCN Models and their predictions for all sheets
        
    multivariateTCNModels = []

    scaledDFs = scaledDF
    TSs = scaledTS
    
    for index, ts in enumerate(TSs):
        train_scaled = ts[:val_cutoff_index]
        val_scaled = ts[val_cutoff_index:]
        
        model_name = "TCN_" + str(index)
        model_tcn = TCNModel(
            input_chunk_length=input_chunk_len,
            output_chunk_length=output_chunk_len,
            n_epochs=TCNEpochs,
            dropout=dropout,
            dilation_base=dilation_base,
            weight_norm=True,
            kernel_size=kernel_size,
            num_filters=num_filters,
            save_checkpoints=True,
            model_name=model_name,
            force_reset=True,
            **generate_torch_kwargs(),
        )
        
        model_tcn.fit(
            series=train_scaled,
            val_series=val_scaled,
        )
        
        print("Finished Training TCN for index:", index)
        multivariateTCNModels.append(model_tcn)
        
        
    # Identifying the VAR residuals, training, and compiling the VAR model

    predictionsTrain = [] # A list of timeSeries, for the training of the VARIMA residual model
    trueTrain = []
    predictionsVal = [] # A list of timeSeries for the validation of VARIMA residuals
    trueVal = []
    predictionRange = len(dataFrames[0]["Year"]) - (input_chunk_len+output_chunk_len)
    
    for index in range(0, len(multivariateTCNModels)):
        model = multivariateTCNModels[index]
                
        predictionsFull = model.predict(predictionRange, TSs[index])
        predictionsTrain.append(predictionsFull[:val_cutoff_index-(input_chunk_len+output_chunk_len)])
        predictionsVal.append(predictionsFull[val_cutoff_index-(input_chunk_len+output_chunk_len):])
        trueTrain.append(TSs[index][(input_chunk_len+output_chunk_len):val_cutoff_index])
        trueVal.append(TSs[index][val_cutoff_index:])
        
    
    # At this point, you now have equal lengths of true and validation residuals in the form of a list of timeseries. The endpoints of the validation set are concurrent with the endpoints of the given data, so forecasting this for N years forward matches with forecasting main data for N years forward.
    
    # For each, come up with a timeSeries of residuals
    
    residualsTrain = []
    for i in range(0, len(sheetNames)): # Difference of TimeSeries gives a new TimeSeries?
        residualsTrain.append(trueTrain[i] - predictionsTrain[i])
        
    
    residualsVal = []
    for i in range(0, len(sheetNames)): # Difference of TimeSeries gives a new TimeSeries?
        residualsVal.append(trueVal[i] - predictionsVal[i])
    

    # Identify the true values (predicted using the existing model, not retrained ones)
    # Identify the correct values (from existing data)
    # Take a matrix difference of the two


    # Train the VAR(IMA) on ts_residuals
    VARIMAResidualModels = []
    for index in range(0, len(sheetNames)):
        model = VAR(pd.concat([residualsTrain[index].to_dataframe(), residualsVal[index].to_dataframe()]))
        results = model.fit()
        VARIMAResidualModels.append([model, results, results.k_ar])
    
    return scalers, scaledDFs, TSs, multivariateTCNModels, VARIMAResidualModels

def forecastSheetTCNResiduals(scalers, scaledDFs, TSs, multivariateTCNModels, VARIMAResidualModels, index, N): # Note that this is for a specific series, not all the different series together, and returns scaled values
    global var_max_residual
    sdf, sts = scaledDFs[index], TSs[index]
    model_tcn = multivariateTCNModels[index]
    lst = VARIMAResidualModels[index]
    VAR_model = lst[0]
    VAR_res = lst[1]
    VAR_k_ar = lst[2]

    # Forecast the main data
    main_forecast = model_tcn.predict(N, sts) # This is a time series, starting from the next year
    
    # Forecast the residuals
    residual_forecast = VAR_res.forecast((sts.to_dataframe()).values[-VAR_k_ar:], steps=N)
    
    A1 = np.array(main_forecast.all_values())
    A2 = np.array(residual_forecast)
    A1.resize(A2.shape)
    
    A2[A2 > var_max_residual] = var_max_residual
    A2[A2 < -var_max_residual] = var_max_residual
    
    # Add them up

    forecast = A1 + A2
    
    # Add to the global set of variables above (dfForecast, tsForecast)
    columnsOfDf = list(dataFrames[index].columns)
    columnsOfDf.remove("Year")
    
    df = pd.DataFrame(forecast, columns=columnsOfDf)

    original_df = pd.read_csv(sheetNames[index]+".csv")
    df.insert(0, "Year", list(range(int(list(original_df["Year"])[-1])+1, int(list(original_df["Year"])[-1])+N+1)), True)
    
    df["Year"] = ["1/1/"+str(int(x)) for x in df["Year"]]
    df["Year"] = pd.to_datetime(df["Year"])
    
    ts = TimeSeries.from_dataframe(df, time_col="Year")
    unscaled_ts = scalers[index].inverse_transform(ts)
    
    return df, unscaled_ts # The dataframe is scaled, time series is unscaled
    
def forecastAllSheetsTCNResiduals(scalers, scaledDFs, TSs, multivariateTCNModels, VARIMAResidualModels, N):
    dfs = []
    unscaled_tss = []
    
    for index in range(0, len(sheetNames)):
        df, unscaled_ts = forecastSheetTCNResiduals(scalers, scaledDFs, TSs, multivariateTCNModels, VARIMAResidualModels, index, N)
        dfs.append(df)
        unscaled_tss.append(unscaled_ts)
    
    return dfs, unscaled_tss
    
def forecastValidationDataFromTrainingDataOnly(scalers, scaledDFs, TSs, multivariateTCNModels, VARIMAResidualModels, index): # Note that this is for a specific series, not all the different series together, and returns scaled values
    global val_cutoff_index
    
    N = len(dataFrames[0]["Year"]) - val_cutoff_index # This gives the number of items in the validation set
    
    sdf, sts = scaledDFs[index], TSs[index]
    model_tcn = multivariateTCNModels[index]
    lst = VARIMAResidualModels[index]
    VAR_model = lst[0]
    VAR_res = lst[1]
    VAR_k_ar = lst[2]

    # Forecast the main data
    main_forecast = model_tcn.predict(N, sts[0:val_cutoff_index]) # This is a time series, starting from the next year
    
    # Forecast the residuals
    residual_forecast = VAR_res.forecast((sts.to_dataframe()[0:val_cutoff_index]).values[-VAR_k_ar:], steps=N)
    
    A1 = np.array(main_forecast.all_values())
    A2 = np.array(residual_forecast)
    A1.resize(A2.shape)
    
    A2[A2 > 0.05] = 0.05
    A2[A2 < -0.05] = 0.05
    
    # Add them up

    forecast = A1 + A2

    
    # Add to the global set of variables above (dfForecast, tsForecast)
    columnsOfDf = list(dataFrames[index].columns)
    columnsOfDf.remove("Year")
    
    df = pd.DataFrame(forecast, columns=columnsOfDf)

    original_df = pd.read_csv(sheetNames[index]+".csv")
    df.insert(0, "Year", list(range(int(list(original_df["Year"])[-N]), int(list(original_df["Year"])[-1])+1)), True)
    
    df["Year"] = ["1/1/"+str(int(x)) for x in df["Year"]]
    df["Year"] = pd.to_datetime(df["Year"])
    
    ts = TimeSeries.from_dataframe(df, time_col="Year")
    unscaled_ts = scalers[index].inverse_transform(ts)
    
    return df, unscaled_ts


def forecastSeriesUsingARIMA(array, N): # For a single series within a sheet, return the predictions in a np array and the mape error (not the model)
    data = array.flatten()
    
    wasDifferenced = False
    if list(ADFTest(alpha=0.05).should_diff(data))[-1]:
        data = np.diff(data)
        data = data[~np.isnan(data)]
        wasDifferenced = True
    
    
    arima = auto_arima(data, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, seasonal=False, error_action="warn", trace=True, stepwise=True, n_fits=50)
    prediction = np.array(pd.DataFrame(arima.predict(n_periods=N)).values)
    
    if wasDifferenced:
        forecast = np.array([list(array)[-1] + np.sum(prediction[0:i]) for i in range(1, len(prediction)+1)])
    else:
        forecast = prediction
    
    return forecast
    # Use the code in the other folder

def forecastSeriesUsingProphet(df, column, N): # For a single series within a sheet, return the predictions in a np array and the mape error (not the model)
    # Use the code in the other folder
    
    print(column)
    df = df[["Year", column]]
    df = df.rename(columns={"Year": "ds", column: "y"})
    
    model = Prophet()
    model.fit(df)
    
    future = []
    for i in range(0, N):
        date = '1/1/'+str(int(list(pd.read_csv(sheetNames[0]+".csv")["Year"])[-1]+1+i))
        future.append([date])
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds'] = pd.to_datetime(future['ds'])
    
    forecast = model.predict(future)
    return forecast["yhat"].values
    
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
def compute_loss(predArray, col): # The lower the better; # It is assumed that the array is from 2020 onwards and extends to atleast 2024 (length >= 5)
    d2023 = float(predArray.flatten()[3])
    d2024 = float(predArray.flatten()[4])
    df2 = pd.read_csv("data_testing.csv")
    t2023 = float(df2[col].values[0])
    t2024 = float(df2[col].values[1])
    
    if (math.isnan(d2023) or math.isnan(t2023)) and (math.isnan(d2024) or math.isnan(t2024)):
        return 0
    
    if math.isnan(d2023) or math.isnan(t2023):
        return mean_absolute_percentage_error([t2024], [d2024])
        
    if math.isnan(d2024) or math.isnan(t2024):
        return mean_absolute_percentage_error([t2023], [d2023])
    
    return mean_absolute_percentage_error([t2023, t2024], [d2023, d2024])
    
    

def identifyBestForecastsFromScratch(index, N): # For all indices, return a list of unscaled data frames, based on the MAPE error.
    # First train all the models
    
    sheetTCNForecasts = []
    colNames = []
    bestPredictions = [] # A list of numpy arrays
    
    # Iterate through and make 5 predictions for the sheet
    for i in range(0, 5):
        scalers, scaledDFs, TSs, multivariateTCNModels, VARIMAResidualModels = generateForecastingModels()
        df, unscaled_ts = forecastSheetTCNResiduals(scalers, scaledDFs, TSs, multivariateTCNModels, VARIMAResidualModels, index, N)
        sheetTCNForecasts.append(unscaled_ts.to_dataframe())
    
    # For each series (excluding the "Year") within the sheet, also make the ARIMA and Prophet predictions
    for col in dataFrames[index].columns:
        if str(col) != "Year":
            colNames.append(col)
            vals = dataFrames[index][[col]].values
            arimaPrediction = forecastSeriesUsingARIMA(vals, N)
            prophetPrediction = forecastSeriesUsingProphet(dataFrames[index], col, N)
            preds = [sheetforecast[col].values.flatten() for sheetforecast in sheetTCNForecasts] + [arimaPrediction, prophetPrediction] # The order matters! If no testing data is found, then the first one is chosen
            losses = [compute_loss(pred, col) for pred in preds]
            best_index = losses.index(min(losses))
            best_pred = preds[losses.index(min(losses))]
##            modelToSave = None
##            
##            if best_index == 0:
##                # This is the TCN Forecast
##                
##                
##            elif best_index == 1:
##                # This is the ARIMA prediction
##                
##                
##            elif best_index == 2:
##                # This is the Prophet prediction
##                
                
                
            bestPredictions.append(best_pred)
    
    bestPredictions = np.array([p.flatten() for p in bestPredictions]).transpose()
    
    df = pd.DataFrame(bestPredictions, columns=colNames)
    original_df = pd.read_csv(sheetNames[index]+".csv")
    df.insert(0, "Year", list(range(int(list(original_df["Year"])[-N]), int(list(original_df["Year"])[-1])+1)), True)
    
    return df
            
            
print("Started Model Training... \nThis may take a while...")
    
    # Select the best predictions for that sheet
    # Add this to a numpy array
    # Turn the numpy array for the entire sheet into a dataframe with the right columns and the right years
    # Add this dataframe to a list
    # Return this list

# Note: there is no need to avoid optimizing specific forecasts, even if it will be overwritten in the future using ANN. When predicting using ANN, the growth parameters will be ignored anyways, so no need to filter here

#print(identifyBestForecastsFromScratch(2, 10))
