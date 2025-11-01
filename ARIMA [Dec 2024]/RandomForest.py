import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import math




def forestRegTrain(x, y, n_est=10):
    reg = RandomForestRegressor(n_estimators=n_est, oob_score=True)
    reg.fit(x, y)
    return reg


##pred = reg.predict([[0.05, 0.02], [0.20, 0.20], [0.8, 0.92]])
##print(pred)
##
##mse = mean_squared_error([1, 2, 3], pred)
##print("MSE:", mse)































### Referred sources (not word-for-word): https://www.geeksforgeeks.org/random-forest-regression-in-python/
