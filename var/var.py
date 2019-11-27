# percentage changes vs log changes
# return vs exponentially weighted returns
import pandas as pd
import numpy as np
from itertools import zip
import matplotlib.pyplot as plt
from math import ceil

# Import stock data
df = pd.DataFrame({'A': 100*np.random.normal(1,0.05,100), 'B': 100*np.random.normal(1.1,0.05,100)})

# Find names of stocks
stocks = df.columns

# Find returns of stocks
for stock in stocks:
    df['return_day_' + stock] = 0
    for i in range(1, len(df[stock])):
        df.iloc[i, df.columns.get_loc('return_day_' + stock)] = (df.iloc[i, df.columns.get_loc(stock)]/df.iloc[i-1, df.columns.get_loc(stock)] - 1)

# Plot returns distributions/histogram
plt.figure()
col_plots = 4
i = 1
for stock in stocks:
    rows = ceil(len(stocks)/col_plots)
    subplot = str(rows) + str(col_plots) + str(i)
    plt.subplot(int(subplot))
    plt.hist(df['return_day_' + stock], bins=25)
    plt.title(stock)
    plt.grid(True)
    i += 1
plt.show()

# returns to convert to array to calculate variance
returns = df.iloc[:, len(stocks) : 2*len(stocks)]
returns = returns.to_numpy()

# std of individual stocks
returns_std_ind = np.std(returns, axis=0, ddof=1)

# std of portfolio of stocks
stock_weights = np.array([0.6, 0.4])
returns_cov_port = np.cov(returns, rowvar=False, ddof=1)
# var(aX + bY) = var(aX) + var(aY) + 2ab cov(XY) = a^2 var(X) + b^2 var(Y) + 2ab cov(XY)
returns_std_port = np.sqrt(stock_weights.T@returns_cov_port@stock_weights)

def percentage_var(std_vector, pct_var):
    var_map = {1: 2.33, 5: 1.64}
    return std_vector*var_map[pct_var]

var_ind = percentage_var(returns_std_ind, 1)
var_port = percentage_var(returns_std_port, 1)