#ARIMA Modeling: Identifying p and q parameters using autocorrelation and partial autocorrelation functions

import make_stationary
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(data_diff, nlags=20)
lag_pacf = pacf(data_diff, nlags=20, method='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_diff)), linestyle = '--', color='gray') #confidence intervals
plt.axhline(y=1.96/np.sqrt(len(data_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_diff)), linestyle = '--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()