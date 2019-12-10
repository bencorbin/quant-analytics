import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self, s, k, t, sigma, r):
        self.s = s
        self.k = k
        self.t = t
        self.sigma = sigma
        # self.d = d
        self.r = r

    def d1(self):

        # return (np.log(self.s / self.k) + (self.r - self.d + 0.5*self.sigma**2)*self.t) / (np.sqrt(self.t)*self.sigma)
        return (np.log(self.s / self.k) + (self.r - + 0.5 * self.sigma ** 2) * self.t) / (np.sqrt(self.t) * self.sigma)

    def d2(self):

        # return (np.log(self.s / self.k) + (self.r - self.d - 0.5*self.sigma**2)*self.t) / (np.sqrt(self.t)*self.sigma)
        return (np.log(self.s / self.k) + (self.r - 0.5 * self.sigma ** 2) * self.t) / (np.sqrt(self.t) * self.sigma)

    def european_call(self):
        """
        Calculate European Call Price based on the Black-Scholes models:
        :return: The price of a European call option via the Black Scholes Model
        """
        d1 = self.d1()
        d2 = self.d2()

        # call_price = (self.s * np.exp(-self.d*self.t) * norm.cdf(d1, 0.0, 1.0)
        #               - self.k * np.exp(-self.r * self.t) * norm.cdf(d2, 0.0, 1.0))

        call_price = (self.s * np.exp(-self.t) * norm.cdf(d1, 0.0, 1.0)
                      - self.k * np.exp(-self.r * self.t) * norm.cdf(d2, 0.0, 1.0))
        return call_price

    def european_put(self):
        """
        Calculate European Put Price based on the Black-Scholes models:
        :return: The price of a European put option via the Black Scholes Model
        """
        d1 = self.d1()
        d2 = self.d2()

        # put_price = (self.k * np.exp(-self.r * self.t) * norm.cdf(-d2, 0.0, 1.0))\
        #             - (self.s * np.exp(-self.d * self.t) * norm.cdf(-d1, 0.0, 1.0))

        put_price = (self.k * np.exp(-self.r * self.t) * norm.cdf(-d2, 0.0, 1.0)) \
                    - (self.s * np.exp(-self.t) * norm.cdf(-d1, 0.0, 1.0))
        return put_price

    def delta(self, option_type):
        """
        Delta measures the sensitivity of the theoretical value of an option to a change in price of the underlying
        stock price.
        :param option_type: this is either "call" or "put"
        :return: the change in the theoretical value of an option for a change in price of the underlying stock price.
        """

        d1 = self.d1()

        if option_type == "call":
            return norm.cdf(d1, 0.0, 1.0)

        elif option_type == "put":

            return norm.cdf(d1, 0.0, 1.0) - 1

    def dNd1_ds(self, plus_or_minus):
        """
        Calculate N(d1) differentiated with respect to S
        :param plus_or_minus: Specify whether d1 or -d1 is integrated i.e. 'plus' for d[N(d1)]/ds or 'minus' for
        d[N(-d1)]/ds
        :return: return N(+/- d1) differentiated with respect to S
        """
        if plus_or_minus == 'plus':
            return np.exp(-0.5 * self.d1()**2)/np.sqrt(2 * np.pi)
        elif plus_or_minus == 'minus':
            return np.exp(-0.5 * (-self.d1())**2)/np.sqrt(2 * np.pi)

    def dNd2_ds(self):
        """
        Calculate N(d2) differentiated with respect to S
        :return: return N(d2) differentiated with respect to S
        """

        return np.exp(-0.5 * self.d2()**2)/np.sqrt(2 * np.pi)

    def gamma(self, option_type):
        """
        Gamma measures the sensitivity of Delta to a change in price of the underlying stock price.
        :param option_type: this is either "call" or "put"
        :return: the change in the sensitivity of Delta for a change in price of the underlying stock price.
        """
        if option_type == 'call' or 'put':
            return self.dNd1_ds('plus')/(self.s * self.sigma * np.sqrt(self.t))
        else:
            raise Exception("variable 'option_type' must be either 'call' or 'put'.")

    def rho(self, option_type):
        """
        Rho measures the sensitivity of the theoretical value of an option to a change in the continuously compounded
        interest rate.
        :param option_type: this is either "call" or "put"
        :return: the change in the theoretical value of an option for a change in the continuously compounded interest
        rate.
        """
        if option_type == "call":
            return self.t * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2(), 0, 1)
        else:
            return -self.t * self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2(), 0, 1)

    def theta(self, option_type):
        """
        Theta measures the sensitivity of the theoretical value of an option to a change in the time to maturity.
        :param option_type: this is either "call" or "put"
        :return: the change in the theoretical value of an option for a change in the time to maturity.
        """
        if option_type == "call":
            return -self.s * self.sigma * self.dNd1_ds('plus') / (2 * np.sqrt(self.t)) \
                   - self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2(), 0, 1)
        else:
            return -self.s * self.sigma * self.dNd1_ds('minus') / (2 * np.sqrt(self.t))\
                   + self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2(), 0, 1)

    def vega(self, option_type):
        """
        Vega measures the sensitivity of the theoretical value of an option to a change in the volatility of return of
        the underlying asset.
        :param option_type: this is either "call" or "put"
        :return: the change in the theoretical value of an option for a change in the time to maturity.
        """

        if option_type == 'call' or 'put':
            return self.s * np.sqrt(self.t) * self.dNd1_ds('plus')
        else:
            raise Exception("variable 'option_type' must be either 'call' or 'put'.")


bs = BlackScholes(154.08, 147, 1 / 365, 0.2331, 0.1)
bs.delta('call')
bs.gamma('call')
bs.rho('call')
bs.theta('call')
bs.vega('call')
