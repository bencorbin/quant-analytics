from numpy import real, exp, sqrt, log, pi, inf
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


class HestonDiffusion:

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

    def diffusion(self, timesteps, simulations):
        dt = self.t / timesteps

        s = np.zeros((simulations, timesteps))
        s[:, 0] = np.array([self.s0] * simulations)

        v = np.zeros((simulations, timesteps))
        v[:, 0] = np.array([self.v0] * simulations)

        z1 = np.random.randn(simulations, timesteps)
        z2 = np.random.randn(simulations, timesteps)

        z2 = self.rho * z1 + sqrt(1 - self.rho ** 2) * z2

        for step in range(0, timesteps - 1):
            v[:, step + 1] = v[:, step] + self.kappa * (self.theta - v[:, step]) * dt + \
                             self.sigma * sqrt(v[:, step]) * z1[:, step] * sqrt(dt)
            v[:, step + 1] = v[:, step + 1] * (v[:, step + 1] > 0)

            s[:, step + 1] = s[:, step] + s[:, step] * (self.r * dt + sqrt(v[:, step]) * z2[:, step] * sqrt(dt))

        mean_s = s.mean(axis=0)
        mean_v = v.mean(axis=0)

        plt.figure()
        plt.plot(np.arange(0, self.t * 365, dt * 365), mean_s, label='Stock')
        plt.legend()
        plt.xlabel('Time (Days)')
        plt.ylabel("Stock Price")
        plt.title("Heston Stock Price Diffusion")
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(np.arange(0, self.t * 365, dt * 365), mean_v, label='Volatility')
        plt.legend()
        plt.xlabel('Time (Days)')
        plt.ylabel("Volatility")
        plt.title("Heston Volatility Diffusion")
        plt.grid()
        plt.show()


hest = HestonDiffusion(154.08, 0.0105, 0.0837, 74.32, 3.4532, 0.1, -0.8912, 15 / 365, 147)
hest.diffusion(200, 10000)
