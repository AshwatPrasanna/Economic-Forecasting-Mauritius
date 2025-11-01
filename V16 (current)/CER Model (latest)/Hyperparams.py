import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler

val_cutoff_index = 20 # Everything including and beyond this is considered validation data
input_chunk_len = 3
output_chunk_len = 2
TCNEpochs = 500
dropout = 0.25
dilation_base = 2
kernel_size = 2
num_filters = 3
forecast_time_step = 10 # Must be larger than the length of both training and validation data; consider MODIFYing this later to be immediate (time_step = 1)
ann_weightage = 0.2
var_max_residual = 0.05 # This is in scaled units
sheetNames = ["data_labor", "data_indices", "data_sectoral", "data_tax", "data_trade", "data_price", "data_pubfin"]



########## The below variables are auto-generated from the files ##########

dataFrames = [pd.read_csv(name + ".csv") for name in sheetNames] # This is not to be modified for any reason
for df in dataFrames:
    df["Year"] = pd.to_datetime(["1/1/"+str(int(year)) for year in df["Year"]])

timeSeries = [TimeSeries.from_dataframe(df, time_col="Year") for df in dataFrames]
