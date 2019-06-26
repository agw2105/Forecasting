#Grid search for SARIMAX parameters

import make_stationary
import itertools

p = d = q = range(0,2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for seasonal ARIMA:')
print('SARIMAX {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX {} x {}'.format(pdq[2], seasonal_pdq[4]))

#using stationary data
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data_diff, order = param,
                                            seasonal_order = param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA {} x {} 12 - AIC: {}'.format(param, param_seasonal, results.aic))
        except:
            continue
