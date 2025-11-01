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
import logging
logging.disable(logging.CRITICAL)


########################################################################

def read_csv_with_time(year, csv): # Reads a CSV and auto-sets the year as the time variable.
    df = pd.read_csv(csv)
    df[year] = ["1/1/"+str(int(x)) for x in df[year]]
    df[year] = pd.to_datetime(df[year])
    return df

sheetNames = ["data_labor", "data_indices", "data_sectoral", "data_tax", "data_trade", "data_price", "data_pubfin"]
dataFrames = [read_csv_with_time("Year", (name + ".csv")) for name in sheetNames]
#dataFramesForecast_no_residuals = []
scalers = []
#dataFramesForecast = []
multivariateTCNModels = [] # These function on scaled data
VARIMAResidualModels = []

timeSeries = [TimeSeries.from_dataframe(df, time_col="Year") for df in dataFrames]
#timeSeriesForecast_no_residuals = []
#timeSeriesForecast = []

########################################################################

def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }
    
def scaleDataFramesAndTimeSeries(): # Takes the existing dataframes and time series and returns a scaled version, updating the list of scalers, global variable. It does NOT update the global lists of time series or data frames
    global dataFrames
    global timeSeries
    global sheetNames
    global scalers
    
    scaledDF = []
    scaledTS = []
    for index in range(0, len(sheetNames)):
        scaler = Scaler()
        sts = scaler.fit_transform(timeSeries[index])
        scaledTS.append(sts)
        scalers.append(scaler)
        sdf = sts.to_dataframe()
        scaledDF.append(sdf)
    return scaledDF, scaledTS
        

def fitTCNModelToSheets():
    global val_cutoff_index
    global input_chunk_len
    global output_chunk_len
    global TCNEpochs
    global dropout
    global dilation_base
    global kernel_size
    global num_filters

    if len(multivariateTCNModels) != 0:
        return multivariateTCNModels

    scaledDFs, TSs = scaleDataFramesAndTimeSeries() # List of all DataFrames and Time Series
    
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
        
        #model_tcn = model_tcn.load_from_checkpoint(model_name=("TCN_" + str(index)), best=True)
        
        model_tcn.fit(
            series=train_scaled,
            val_series=val_scaled,
        )
        
        print("Finished Training TCN for index:", index)
        multivariateTCNModels.append(model_tcn)
        
    return multivariateTCNModels



def determineResiduals():
    global val_cutoff_index

    scaledDFs, TSs = scaleDataFramesAndTimeSeries()

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
    return residualsTrain, residualsVal # This is in the form of a timeseries

def VARIMAResiduals():

    residualsTrain, residualsVal = determineResiduals() # List of TimeSeries, each of which will be forecasted using VARIMA

    # Train the VAR(IMA) on ts_residuals
    for index in range(0, len(sheetNames)):

        model = VAR(pd.concat([residualsTrain[index].to_dataframe(), residualsVal[index].to_dataframe()]))
        results = model.fit()
        VARIMAResidualModels.append([model, results, results.k_ar])
    
    return VARIMAResidualModels
    
def forecastSeries(index, N): # Note that this is for a specific series, not all the different series together, and returns scaled values
    scaledDFs, TSs = scaleDataFramesAndTimeSeries()
    sdf, sts = scaledDFs[index], TSs[index]
    fitTCNModelToSheets()
    model_tcn = multivariateTCNModels[index]
    VARIMAResiduals() # This generates the VAR models
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
    
    A2[A2 > 0.2] = 0.2
    A2[A2 < -0.2] = 0.2
    
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
    
    return df, unscaled_ts
    
    
def forecastValidationDataFromTrainingDataOnly(index): # Note that this is for a specific series, not all the different series together, and returns scaled values
    N = len(dataFrames[0]["Year"]) - val_cutoff_index # This gives the number of items in the validation set
    scaledDFs, TSs = scaleDataFramesAndTimeSeries()
    sdf, sts = scaledDFs[index], TSs[index]
    fitTCNModelToSheets()
    model_tcn = multivariateTCNModels[index]
    VARIMAResiduals() # This generates the VAR models
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

########################################################################

scaledDF, scaledTS = scaleDataFramesAndTimeSeries()
df, udf = forecastSeries(1, 10)
#
scaledTS[1].plot()
TimeSeries.from_dataframe(df, time_col="Year").plot()
#plt.show()
#
#print(udf)


forecastSeries(0, 10)
