import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.DataFrame({"Stock_1": [1, 2, 3, 4, 5], "stock_2": [2, 3, 4, 5, 6]})

stocks = df.to_numpy()
# ~~~~~~~~~~~~ x = stock1 ~~~~~~~~~~~~
X = stocks[:, 0].reshape(-1, 1)

# ~~~~~~~~~~~~ y = log(stock2) - spread ~~~~~~~~~~~~
y = np.log(stocks[:, 1]) - (stocks[:, 1] - stocks[:, 0])
y = y.reshape(-1, 1)

# ~~~~~~~~~~~~ linear regression - ordinary least squares ~~~~~~~~~~~~
reg = LinearRegression().fit(X, y)
coefficient = reg.coef_
intercept = reg.intercept_

# ~~~~~~~~~~~~ spread = -log(stock2) - spread ~~~~~~~~~~~~
spread = -np.log(stocks[:, 1]).reshape(-1, 1) + coefficient * X + intercept

plt.figure()
plt.plot(np.arange(0, 5, 6), spread.reshape(1, -1))
# fig.plot(stocks[:, 1] - stocks[:, 0])
plt.show()


class PairSelection:

    def __init__(self):
        self.df = df

    def mse(self):
        # TODO: update so that it works for more than 2 stocks

        self.df["diff"] = self.df.iloc[:, 0] - self.df.iloc[:, 1]

        mean = self.df["diff"].mean()
        variance = self.df["diff"].var()

        sum = 0
        for index, row in df.iterrows():
            sum = sum + ((row["diff"] - mean)/variance)**2

        mse = sum/len(self.df.index)

        return mse

    def cointegration(self):

        return passsed_test



