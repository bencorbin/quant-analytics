from heston.closed_form import Heston
from numpy import arange
import matplotlib.pyplot as plt


class HestonGreeks:

    def __init__(self, greeks, *initial_conditions):
        self.greeks = greeks

        (self.s0, self.v0, self.theta, self.kappa, self.sigma, self.r, self.rho, self.t, self.k) = initial_conditions

    def greek_function_mapping(self, hest, greek):
        map = {
            'delta': hest.delta(),
            'gamma': hest.gamma(),
            'rho': hest.rho_h(),
            'theta': hest.theta_h(),
            'vega': hest.vega(),
            'volga': hest.volga(),
            'vanna': hest.vanna()
        }

        return map[greek]

    def greek_sim(self):

        for greek in self.greeks['greeks']:
            store_greek = []
            start = self.greeks['start']
            end = self.greeks['end']
            for stock in arange(start, end, 0.1):
                hest = Heston(stock, self.v0, self.theta, self.kappa, self.sigma, self.r, self.rho, self.t, self.k)
                greek_value = self.greek_function_mapping(hest, greek)
                store_greek.append(greek_value)

            plt.figure()
            plt.plot(arange(start, end, 0.1), store_greek)
            plt.xlabel('Stock Price')
            plt.ylabel("{0}".format(greek))
            plt.title("{0} of European Call Option".format(greek))
            plt.grid()
            plt.show()


initial_conditions = [154.08, 0.0105, 0.0837, 74.32, 3.4532, 0.1, -0.8912, 1 / 365, 147]
# plot = HestonGreeks({'start': 143, 'end': 150, 'greeks': ['delta', 'gamma', 'rho', 'theta', 'vega', 'volga', 'vanna']},
#                     *initial_conditions)
plot.greek_sim()



