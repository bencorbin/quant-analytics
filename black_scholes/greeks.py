import numpy as np
from scipy.stats import norm

class greeks:
    def __init__(self, s, k, t, sigma, d, r):
        self.s = s
        self.k = k
        self.t = t
        self.sigma = sigma
        self.d = d
        self.r = r

    def greek_delta(self, option_type):
        if option_type == "call":
            d1 = (np.log(self.s / self.k) + (self.r - self.d + 0.5 * self.sigma ** 2) * self.t) / (
                    np.sqrt(self.t) * self.sigma)

            return norm.cdf(d1, 0.0, 1.0)

        elif option_type == "put":
            d1 = (np.log(self.s / self.k) + (self.r - self.d + 0.5 * self.sigma ** 2) * self.t) / (
                    np.sqrt(self.t) * self.sigma)

            return norm.cdf(d1, 0.0, 1.0) - 1