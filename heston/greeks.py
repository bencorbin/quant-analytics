from heston.closed_form import Heston
from black_scholes.closed_form import BlackScholes
from numpy import arange
import matplotlib.pyplot as plt


class HestonGreeks:

    def __init__(self, greeks, heston_init_conditions, blackscholes_init_conditions):
        self.greeks = greeks

        (self.s0, self.v0, self.theta, self.kappa, self.sigma, self.r, self.rho, self.t, self.k) = heston_init_conditions

        (self.s, self.k, self.t, self.sigma, self.r) = blackscholes_init_conditions

    def hest_greek_function_mapping(self, hest, greek):
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

    def bs_greek_function_mapping(self, bs, greek, option_type):
        map = {
            'delta': bs.delta(option_type),
            'gamma': bs.gamma(option_type),
            'rho': bs.rho(option_type),
            'theta': bs.theta(option_type),
            'vega': bs.vega(option_type),
            # 'volga': bs.volga(option_type),
            # 'vanna': bs.vanna(option_type)
        }

        return map[greek]

    def greek_sim(self):

        for greek in self.greeks['greeks']:
            store_hest_greek = []
            store_bs_greek = []
            start = self.greeks['start']
            end = self.greeks['end']
            for stock in arange(start, end, 0.5):
                hest = Heston(stock, self.v0, self.theta, self.kappa, self.sigma, self.r, self.rho, self.t, self.k)
                bs = BlackScholes(stock, self.k, self.t, self.sigma, self.r)

                hest_greek_value = self.hest_greek_function_mapping(hest, greek)
                bs_greek_value = self.bs_greek_function_mapping(bs, greek, 'call')

                store_hest_greek.append(hest_greek_value)
                store_bs_greek.append(bs_greek_value)

            plt.figure()
            plt.plot(arange(start, end, 0.5), store_hest_greek, arange(start, end, 0.5), store_bs_greek)
            plt.xlabel('Stock Price')
            plt.ylabel("{0}".format(greek))
            plt.title("{0} of European Call Option".format(greek))
            plt.grid()
            plt.show()


hest_initial_conditions = (154.08, 0.0105, 0.0837, 74.32, 3.4532, 0.1, -0.8912, 15 / 365, 155)
bs_initial_conditions = (154.08, 155, 15 / 365, 0.2331, 0.1)
# plot = HestonGreeks({
#     'start': 143,
#     'end': 150,
#     'greeks':
#         ['delta', 'gamma', 'rho', 'theta', 'vega', 'volga', 'vanna']
# },
#     *initial_conditions)
plot = HestonGreeks({'start': 120, 'end': 190, 'greeks': ['delta', 'gamma', 'rho', 'theta']},
                    hest_initial_conditions,
                    bs_initial_conditions)
plot.greek_sim()



