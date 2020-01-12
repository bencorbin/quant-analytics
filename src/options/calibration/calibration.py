from scipy.optimize import minimize
from src.options.heston.closed_form import Heston
from numpy import array
import pandas as pd
import xlrd


class CalibrateHeston:
    def __init__(self, known_parameters, initial_guess, bounds):
        """
            :param v: volatility
            :param theta: long run average volatility (vbar)
            :param kappa: mean reversion (speed) of variance to long run average
            :param sigma: volatility of volatility (vvol)
            :param rho: correlation between the brownian motion of the stock price and the volatility.
        """

        self.known_s = known_parameters['s']
        self.known_r = known_parameters['r']
        self.known_t = known_parameters['t']
        self.known_k = known_parameters['k']
        self.known_c = known_parameters['c']
        self.initial_guess = initial_guess
        self.bounds = bounds

    def objective(self, guess):
        sum_of_relative_difference = 0

        for i in range(0, 14):

            # s0, v0, theta, kappa, sigma, r, rho, t, k:
            heston = Heston(s0=self.known_s[i], v0=guess[0], k=self.known_k[i], t=self.known_t[i], r=self.known_r[i],
                            theta=guess[1], kappa=guess[2], sigma=guess[3], rho=guess[4])

            pred_price = heston.call()
            print("Predicted Price {}".format(pred_price))
            print("Predicted Params {}".format(guess))

            actual_price = self.known_c[i]
            print("Diff {}".format(actual_price - pred_price))
            trade_abs_difference = abs(actual_price - pred_price)
            trade_relative_difference = trade_abs_difference/actual_price

            sum_of_relative_difference = sum_of_relative_difference + trade_relative_difference

            print("Sum of rel difference {}".format(sum_of_relative_difference))

        return sum_of_relative_difference

    def local_optimisation(self):
        result = minimize(self.objective, self.initial_guess, bounds=self.bounds, tol=0.05)
        return result


# s0, v0, theta, kappa, sigma, r, rho, t, k)
known_parameters = pd.read_excel(r"C:\Dev\quant-analytics\src\options\calibration\calibration_test.xlsx", header=0, sheet_name="Sheet2")

# initial_guess(v, theta, kappa, sigma, rho)
initial_guess = array([0.295, 0.9, 0.09, 0.7, -0.2])
bounds = array([(0, 1), (0, 1), (0, 5), (0, 5), (-1, 1)])

cali = CalibrateHeston(known_parameters, initial_guess, bounds)

cali.objective(initial_guess)
output = cali.local_optimisation()


