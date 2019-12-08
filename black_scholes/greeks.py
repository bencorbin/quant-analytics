import numpy as np
from scipy.stats import norm
from black_scholes.closed_form import BlackScholes


class Greeks:

    def __init__(self, s, k, t, sigma, d, r):
        self.s = s
        self.k = k
        self.t = t
        self.sigma = sigma
        self.d = d
        self.r = r

    def greek_delta(self, option_type):

        d1 = (np.log(self.s / self.k) + (self.r - self.d + 0.5 * self.sigma ** 2) * self.t) / (
                np.sqrt(self.t) * self.sigma)

        if option_type == "call":
            return norm.cdf(d1, 0.0, 1.0)

        elif option_type == "put":

            return norm.cdf(d1, 0.0, 1.0) - 1

    def greek_gamma(self, option_type):
        """
        :param option_type: this is either "call" or "put"
        :return: Gamma which measures delta's rate of change
        """
        if option_type == "call":
            d1 = self.d1_european_call()

        elif option_type == "put":
            d1 = self.d1_european_put()

        gamma = si.norm.pdf(d1) / (self.s * self.sigma * np.sqrt(self.t))

        return gamma

    def greek_vega(self, option_type):
        """
        :param option_type: this is either "call" or "put"
        :return: Vega which Measures Impact of a Change in Volatility
        """
        if option_type == "call":
            d1 = self.d1_european_call()

        elif option_type == "put":
            d1 = self.d1_european_put()

        vega = self.s * si.norm.pdf(d1) * np.sqrt(self.t)
        return vega

    def greek_rho(self, option_type):
        if option_type == "call":
            rho = self.k * self.t * np.exp(-self.r * self.t) * si.norm.cdf(self.d2_european_call())

        elif option_type == "put":
            rho = -self.k * self.t * np.exp(-self.r * self.t) * si.norm.cdf(-self.d2_european_call())
        return rho

    def greek_theta(self, option_type):
        if option_type == "call":
            theta = (-np.exp(-self.d * self.t)) * (self.s * si.norm.pdf(self.d1_european_call()) * self.sigma) / \
                    2 * np.sqrt(self.t) \
                    - self.r * self.k * (np.exp(-self.r * self.t)) * si.norm.cdf(self.d2_european_call()) \
                    + self.d * self.s * (np.exp(-self.d * self.t)) * si.norm.cdf(self.d1_european_call())

        elif option_type == "put":
            theta = (-np.exp(-self.d * self.t)) * (self.s * si.norm.pdf(self.d1_european_put()) * self.sigma) / \
                    2 * np.sqrt(self.t) \
                    + self.r * self.k * (np.exp(-self.r * self.t)) * si.norm.cdf(- self.d2_european_put()) \
                    - self.d * self.s * (np.exp(-self.d * self.t)) * si.norm.cdf(- self.d1_european_put())

        return theta
