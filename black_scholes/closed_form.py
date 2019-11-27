import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self, s, k, t, sigma, d, r):
        self.s = s
        self.k = k
        self.t = t
        self.sigma = sigma
        self.d = d
        self.r = r

    def european_call(self):
        """
        Calculate European Call Price based on the Black-Scholes models:
            The formula can be found here: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
        :function si.norm.cdf: The cumulative distribution function of the standard normal distribution
        :param S:       Stock price of the underlying security
        :param K:       The strike price of the option, also frequently called exercise price.
        :param T:       Time to maturity in years.
        :param Sigma:   The volatility of returns of the underlying asset
        :param d:       The dividend yield. An annual rate, expressed in terms of continuous compounding.
        :param r:       The risk free rate. An annual rate, expressed in terms of continuous compounding.
        :return:        The price of a European call option via the Black Scholes Model
        """
        d1 = (np.log(self.s / self.k) + (self.r - self.d + 0.5*self.sigma**2)*self.t) / (np.sqrt(self.t)*self.sigma)
        d2 = (np.log(self.s / self.k) + (self.r - self.d - 0.5*self.sigma**2)*self.t) / (np.sqrt(self.t)*self.sigma)

        call_price = (self.s * np.exp(-self.d*self.t) * norm.cdf(d1, 0.0, 1.0)
                      - self.k * np.exp(-self.r * self.t) * norm.cdf(d2, 0.0, 1.0))
        return call_price

    def european_put(self):
        """
        Calculate European Put Price based on the Black-Scholes models:
            The formula can be found here: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
        :function si.norm.cdf: The cumulative distribution function of the standard normal distribution
        :param S:       Stock price of the underlying security
        :param K:       The strike price of the option, also frequently called exercise price.
        :param T:       Time to maturity in years.
        :param Sigma:   The volatility of returns of the underlying asset
        :param d:       The dividend yield. An annual rate, expressed in terms of continuous compounding.
        :param r:       The risk free rate. An annual rate, expressed in terms of continuous compounding.
        :return:        The price of a European put option via the Black Scholes Model
        """
        d1 = (np.log(self.s / self.k) + (self.r - self.d + 0.5 * self.sigma ** 2) * self.t) / \
             (np.sqrt(self.t) * self.sigma)
        d2 = (np.log(self.s / self.k) + (self.r - self.d - 0.5 * self.sigma ** 2) * self.t) / \
             (np.sqrt(self.t) * self.sigma)

        put_price = (self.k * np.exp(-self.r * self.t) * norm.cdf(-d2, 0.0, 1.0))\
                    - (self.s * np.exp(-self.d * self.t) * norm.cdf(-d1, 0.0, 1.0))
        return put_price
