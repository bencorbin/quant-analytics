import numpy as np
from scipy import integrate


class Heston:

    def __init__(self, s0, v0, theta, kappa, sigma, r, rho, t, k):
        """
        :param s0: Initial stock price
        :param v0: Initial volatility
        :param theta: long run variance
        :param kappa: Mean reversion of variance
        :param sigma: Volatility of volatility
        :param r: Risk-free rate
        :param rho: Correlation
        :param t: Time to maturity
        :param k: Exercise price
        """
        self.s0 = s0
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.r = r
        self.rho = rho
        self.t = t
        self.k = k

    def char_func(self, phi, j):
        """
        Calculate the characteristic function solution to the Heston PDE
        :param phi:
        :param j:
        :return:
        """

        if j == 1:
            u = 0.5
            b = self.kappa - self.rho*self.sigma
        else:
            u = -0.5
            b = self.kappa

        a = self.kappa*self.theta
        x = np.log(self.s0)

        d = np.sqrt((self.rho*self.sigma*phi*complex(0, 1) - b)**2 - self.sigma**2 * (2*u*phi*complex(0, 1) - phi**2))

        # g = (b - self.rho * self.sigma * phi * complex(0, 1) + d)/
        # b - self.rho * self.sigma * phi * complex(0, 1) - d)
        g = (b - self.rho * self.sigma * phi * complex(0, 1) - d) / \
            (b - self.rho * self.sigma * phi * complex(0, 1) + d)

        # c = self.r*phi*complex(0, 1)*self.t + a/self.sigma**2 * ((b-self.rho*self.sigma*phi*complex(0, 1)+d)*self.t -
        # 2*np.log((1-g*np.exp(d*self.t))/(1-g)))
        c = self.r * phi * complex(0, 1) * self.t + a / self.sigma ** 2 * \
            ((b - self.rho * self.sigma * phi * complex(0, 1) - d) * self.t - 2 * np.log((1 - g * np.exp(-d * self.t)) / (1 - g)))

        # d = (b - self.rho * self.sigma * phi * complex(0, 1) + d) /
        # self.sigma ** 2 * ((1 - np.exp(d * self.t)) / (1 - g * np.exp(d * self.t)))
        d = (b-self.rho*self.sigma*phi*complex(0, 1)-d) / \
            self.sigma**2 * ((1-np.exp(-d*self.t))/(1-g*np.exp(- d*self.t)))

        return np.exp(c + d*self.v0 + complex(0, 1)*phi*x)

    def integrand(self, phi, j):

        f = self.char_func(phi, j)

        return np.real(np.exp(-phi*complex(0, 1)*np.log(self.k))*f/(phi*complex(0, 1)))

    def prob_func(self, j):

        y = integrate.quad(self.integrand, 0, np.inf, epsabs=0, args=j, full_output=0)

        return 0.5 + (1/np.pi) * y[0]

    def call(self):

        return self.s0*self.prob_func(1) - self.k * np.exp(-self.r*self.t) * self.prob_func(2)

    def put(self):

        return self.k * self.prob_func(2) - self.s0 * self.prob_func(1)

    def delta_integrand(self, phi):

        return np.real(np.exp(complex(0, -1) * phi * np.log(self.k)) * ((1 - complex(0, 1)/phi) *
                    self.char_func(phi, 1) - self.k * np.exp(-self.r * self.t) / self.s0 * self.char_func(phi, 2)))

    def delta(self):

        y = integrate.quad(self.delta_integrand, 0, np.inf, epsabs=0, full_output=0)

        return 1/2 + 1/np.pi * y[0]


hest = Heston(154.08, 0.0105, 0.0837, 74.32, 3.4532, 0.1, -0.8912, 1/365, 147)
hest.call()
hest.delta()


