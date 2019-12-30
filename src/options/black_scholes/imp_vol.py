from numpy import sqrt, abs, log
from src.options.black_scholes.closed_form import BlackScholes


class ImpVol:

    def __init__(self, s, k, t, r, value, tol, max_iter):
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.value = value
        # self.d = d
        self.tol = tol
        self.max_iter = max_iter

    def sigma_hat(self):
        return sqrt(2 * abs((log(self.s / self.k) + self.r * self.t) / self.t))

    def imp_vol(self):
        price_diff = 1
        iters = 1
        sigma = self.sigma_hat()
        while price_diff >= self.tol & iters < self.max_iter:
            # price of call option
            bs = BlackScholes(self.s, self.k, self.t, sigma, self.r)
            call_price = bs.european_call()
            # find vega
            vega = bs.vega("call")
            increment = (call_price - self.value)/vega
            # new sigma
            sigma = sigma - increment
            # increase iterations by 1
            iters = iters + 1

            price_diff = abs(increment)

        return sigma
