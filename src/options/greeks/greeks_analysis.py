from src.options.heston.closed_form import Heston
from src.options.black_scholes.closed_form import BlackScholes
from numpy import arange
import matplotlib.pyplot as plt


class HestonGreeks:

    def __init__(self, input_dictionary):
        self.greeks = input_dictionary['greeks']
        # TODO: Check if v0 in Heston = sigma in Black-Scholes
        (self.s0, self.v0, self.theta, self.kappa, self.sigma, self.r, self.rho, self.t, self.k) = input_dictionary['heston initial conditions']

        (self.s0, self.k, self.t, self.sigma, self.r) = input_dictionary['black scholes initial conditions']

    def hest_greek_function_mapping(self, hest, greek, option_type):
        # TODO: update so only required greek is calculated when function is called
        map = {
            'delta': hest.delta(option_type),
            'gamma': hest.gamma(option_type),
            'rho': hest.rho_h(option_type),
            'theta': hest.theta_h(option_type),
            'vega': hest.vega(option_type),
            'volga': hest.volga(option_type),
            'vanna': hest.vanna(option_type)
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

    def vary_parameter_mapping(self, param_to_vary, param):

        if param_to_vary == 'Stock Price':
            self.s0 = param
        elif param_to_vary == 'Initial Volatility':
            self.v0 = param
        elif param_to_vary == 'Long Run Variance':
            self.theta = param
        elif param_to_vary == 'Mean Reversion of Variance':
            self.kappa = param
        elif param_to_vary == 'Volatility of Volatility':
            self.sigma = param
        elif param_to_vary == 'Interest Rate':
            self.r = param
        elif param_to_vary == 'Stock/Vol Correlation':
            self.rho = param
        elif param_to_vary == 'Time':
            self.t = param
        elif param_to_vary == 'Exercise Price':
            self.k = param

    def greek_sim(self):

        for greek in self.greeks:
            store_hest_greek = []
            store_bs_greek = []
            start = self.greeks[greek]['start']
            end = self.greeks[greek]['end']
            step = self.greeks[greek]['step']
            option_type = self.greeks[greek]['option_type']
            param_to_vary = self.greeks[greek]['vary_parameter']
            for param in arange(start, end, step):
                self.vary_parameter_mapping(param_to_vary, param)
                hest = Heston(self.s0, self.v0, self.theta, self.kappa, self.sigma, self.r, self.rho, self.t, self.k)
                bs = BlackScholes(self.s0, self.k, self.t, self.sigma, self.r)

                hest_greek_value = self.hest_greek_function_mapping(hest, greek, option_type)
                bs_greek_value = self.bs_greek_function_mapping(bs, greek, option_type)

                store_hest_greek.append(hest_greek_value)
                store_bs_greek.append(bs_greek_value)

            plt.figure()
            plt.plot(arange(start, end, step), store_hest_greek, label='Heston')
            plt.plot(arange(start, end, step), store_bs_greek, label='Black-Scholes')
            plt.legend()
            plt.xlabel('{0}'.format(param_to_vary))
            plt.ylabel("{0}".format(greek))
            plt.title("{0} of European {1} Option".format(greek, option_type))
            plt.grid()
            plt.show()


hest_initial_conditions = (154.08, 0.0105, 0.0837, 74.32, 3.4532, 0.1, -0.8912, 15 / 365, 155)
bs_initial_conditions = (154.08, 155, 15 / 365, 0.2331, 0.1)

plot = HestonGreeks({
    'heston initial conditions': hest_initial_conditions,
    'black scholes initial conditions': bs_initial_conditions,
    'greeks': {
        'delta': {
            'vary_parameter': 'Stock Price',
            'start': 120,
            'end': 190,
            'step': 1,
            'option_type': 'call'
        },
        'vega': {
            'vary_parameter': 'Initial Volatility',
            'start': 0.01,
            'end': 0.1,
            'step': 0.01,
            'option_type': 'call'
        }
    }
 }
)

plot.greek_sim()



