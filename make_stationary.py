#make data stationary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from DickeyFuller import test_stationarity

data = pd.read_csv("data.csv") #import timeseries data
test_stationarity(data) #test for stationarity

#Differencing to make data stationary
data_log = np.log(data)
data_diff = data_log - data_log.shift()
data_diff = data_diff.fillna(0) #first value is NaN due to shift function. The differenced log values should be close to zero, so impute here
plt.plot(data_diff)
plt.show()

test_stationarity(data_diff)

#Decomposition to make data stationary
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = decomposition.plot()
plt.show()

residual.dropna(inplace=True)
test_stationarity(residual)