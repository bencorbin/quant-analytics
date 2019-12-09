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

        d1 = self.d1()

        if option_type == "call":
            return norm.cdf(d1, 0.0, 1.0)

        elif option_type == "put":

            return norm.cdf(d1, 0.0, 1.0) - 1

    def dNd1_ds(self):

        return np.exp(-0.5 * self.d1()**2)/np.sqrt(2 * np.pi)

    def dNd2_ds(self):

        return np.exp(-0.5 * self.d2()**2)/np.sqrt(2 * np.pi)

    def gamma(self, option_type):
        """
        :param option_type: this is either "call" or "put"
        :return: Gamma which measures delta's rate of change
        """
        if option_type == "call":

            return self.dNd1_ds()/(self.s * self.sigma * np.sqrt(self.t))

        # elif option_type == "put":
        #     d1 = self.d1_european_put()
        #
        # gamma = si.norm.pdf(d1) / (self.s * self.sigma * np.sqrt(self.t))

    def rho(self, option_type):
        if option_type == "call":

            return self.t * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2(), 0, 1)

        # elif option_type == "put":
        #     rho = -self.k * self.t * np.exp(-self.r * self.t) * si.norm.cdf(-self.d2_european_call())
        # return rho

    def theta(self, option_type):
        if option_type == "call":

            return -self.s * self.sigma * self.dNd1_ds() / (2 * np.sqrt(self.t)) - \
                   self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2(), 0, 1)

        # elif option_type == "put":
        #     theta = (-np.exp(-self.d * self.t)) * (self.s * si.norm.pdf(self.d1_european_put()) * self.sigma) / \
        #             2 * np.sqrt(self.t) \
        #             + self.r * self.k * (np.exp(-self.r * self.t)) * si.norm.cdf(- self.d2_european_put()) \
        #             - self.d * self.s * (np.exp(-self.d * self.t)) * si.norm.cdf(- self.d1_european_put())
        #
        # return theta

    def vega(self, option_type):
        """
        :param option_type: this is either "call" or "put"
        :return: Vega which Measures Impact of a Change in Volatility
        """
        if option_type == "call":

            return self.s * np.sqrt(self.t) * self.dNd1_ds()

        # elif option_type == "put":
        #     d1 = self.d1_european_put()
        #
        # vega = self.s * si.norm.pdf(d1) * np.sqrt(self.t)
        # return vega


bs = BlackScholes(154.08, 147, 1 / 365, 0.2331, 0.1)
bs.delta('call')
bs.gamma('call')
bs.rho('call')
bs.theta('call')
bs.vega('call')
