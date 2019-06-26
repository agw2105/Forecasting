#Test for Stationarity
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    #determine rolling statistics
    rollmean = timeseries.rolling(12).mean()
    rollstd = timeseries.rolling(12).std()
    
    #plot rolling statistics
    orig = plt.plot(timeseries, color="blue", label="Original")
    mean = plt.plot(rollmean, color="red", label="Rolling Mean")
    std = plt.plot(rollstd, color='black', label = "Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean and Standard Deviation")
    plt.show(block=False)
    
    #Perform Dickey-Fuller test
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag = 'AIC')
    df_output = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '# Lags Used', '# Observations Used'])
    
    for key, value in dftest[4].items():
        df_output['Critical Value (%s)' %key] = value
    print(df_output)
    