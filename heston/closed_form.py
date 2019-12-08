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

        g = (b - self.rho * self.sigma * phi * complex(0, 1) - d) / \
            (b - self.rho * self.sigma * phi * complex(0, 1) + d)

        C = self.r * phi * complex(0, 1) * self.t + a / self.sigma ** 2 * \
            ((b - self.rho * self.sigma * phi * complex(0, 1) - d) * self.t - 2 * np.log((1 - g * np.exp(-d * self.t)) / (1 - g)))

        D = (b-self.rho*self.sigma*phi*complex(0, 1)-d) / \
            self.sigma**2 * ((1-np.exp(-d*self.t))/(1-g*np.exp(- d*self.t)))

        f = np.exp(C + D*self.v0 + complex(0, 1)*phi*x)

        return a, b, d, g, C, D, f

    def integrand(self, phi, j):

        (a, b, d, g, C, D, f) = self.char_func(phi, j)

        return np.real(np.exp(-phi*complex(0, 1)*np.log(self.k))*f/(phi*complex(0, 1)))

    def prob_func(self, j):

        y = integrate.quad(self.integrand, 0, np.inf, epsabs=0, args=j, full_output=0)

        return 1/2 + 1/np.pi * y[0]

    def call(self):

        return self.s0*self.prob_func(1) - self.k * np.exp(-self.r*self.t) * self.prob_func(2)

    def put(self):

        return self.k * self.prob_func(2) - self.s0 * self.prob_func(1)

    def delta_integrand(self, phi):

        (a, b, d, g, C, D, f_1) = self.char_func(phi, 1)
        (a, b, d, g, C, D, f_2) = self.char_func(phi, 2)

        return np.real(np.exp(complex(0, -1) * phi * np.log(self.k)) * ((1 - complex(0, 1)/phi) *
                    f_1 - self.k * np.exp(-self.r * self.t) / self.s0 * f_2))

    def delta(self):

        y = integrate.quad(self.delta_integrand, 0, np.inf, epsabs=0, full_output=0)

        return 1/2 + 1/np.pi * y[0]

    def gamma_integrand(self, phi):

        (a, b, d, g, C, D, f_1) = self.char_func(phi, 1)
        (a, b, d, g, C, D, f_2) = self.char_func(phi, 2)

        return np.real(np.exp(complex(0, -1) * phi * np.log(self.k)) * (1/self.s0 * (1 + complex(0, 1)/phi) *
                    f_1 + self.k * np.exp(-self.r * self.t) / self.s0**2 * (1 - complex(0, 1) *
                    phi) * f_2))

    def gamma(self):

        y = integrate.quad(self.gamma_integrand, 0, np.inf, epsabs=0, full_output=0)

        return 1/np.pi * y[0]

    def rho_integrand(self, phi):

        (a, b, d, g, C, D, f_1) = self.char_func(phi, 1)
        (a, b, d, g, C, D, f_2) = self.char_func(phi, 2)

        return np.real(np.exp(complex(0, -1) * phi * np.log(self.k)) * (self.s0 * f_1 -
                    self.k * np.exp(-self.r * self.t) * (complex(0, 1)/phi + 1) * f_2))

    def rho_h(self):

        y = integrate.quad(self.rho_integrand, 0, np.inf, epsabs=0, full_output=0)

        return 1/2 * self.k * self.t * np.exp(-self.r * self.t) + self.t / np.pi * y[0]

    def dC_dt(self, phi, j):

        (a, b, d, g, C, D, f) = self.char_func(phi, j)

        return -self.r * phi * complex(0, -1) + a/self.sigma**2 * \
               (-(b - self.rho * self.sigma * phi * complex(0, 1) + d) -
                2 * g * d * np.exp(d * self.t) / (1 - g * np.exp(d * self.t)))

    def dD_dt(self, phi, j):

        (a, b, d, g, C, D, f) = self.char_func(phi, j)

        return 1 / self.sigma * (b - self.rho * self.sigma * phi * complex(0, 1) + d) * \
                (d * np.exp(d * self.t) / (1 - g * np.exp(d * self.t)) -
                 g * d * np.exp(d * self.t) * (1 - np.exp(d * self.t))/(1 - g * np.exp(d * self.t)**2))

    def theta_integrand(self, phi):

        (a, b, d, g, C, D, f_1) = self.char_func(phi, 1)
        (a, b, d, g, C, D, f_2) = self.char_func(phi, 2)

        dC_dt_1 = self.dC_dt(phi, 1)
        dC_dt_2 = self.dC_dt(phi, 2)

        dD_dt_1 = self.dC_dt(phi, 1)
        dD_dt_2 = self.dC_dt(phi, 2)

        return np.real(complex(0, -1) * np.exp(complex(0, -1) * phi * np.log(self.k)) / phi *
                       (
                           (dC_dt_1 + self.v0 * dD_dt_1) * f_1 * self.s0 - f_2 * self.k * np.exp(- self.r * self.t) *
                           (
                           self.r + dC_dt_2 + self.v0 * dD_dt_2)
                           )
                       )

    def theta_h(self):

        y = integrate.quad(self.rho_integrand, 0, np.inf, epsabs=0, full_output=0)

        return - self.k * self.r * np.exp(- self.r * self.t) / 2 + 1/np.pi * y[0]


hest = Heston(154.08, 0.0105, 0.0837, 74.32, 3.4532, 0.1, -0.8912, 1/365, 147)
hest.prob_func(1)
hest.call()
hest.delta()
hest.gamma()
hest.rho_h()
hest.theta_h()



