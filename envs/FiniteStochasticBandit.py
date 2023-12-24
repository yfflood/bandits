import numpy as np
from scipy.stats import bernoulli


class FiniteStochasticBandit:
    # accepts a list of K >= 2 floats
    def __init__(self, distributions):
        """ 
        Initialize an instance of an environment class

        Parameters: 
            distributions: list, len = K. 
        """
        self.arms = distributions
        self.history = [] # list of [a, X]
        
        self.means = np.array([distribution.stats(moments='m') for distribution in distributions])
        self.suboptimality_gaps = np.max(self.means) - self.means

    # return the number of arms
    def K(self):
        return len(self.arms)
    

    def pull(self, a):
        """ 
        Pull arm a once, return the realized reward
        
        Parameters:
            a: [0, K-1], index of the chosen arm
        """
        X = self.arms[a].rvs()
        self.history.append([a, X])
        return X


    def regret(self):
        """Calculate the regret incurred so far """
        # TODO: check difference with pseudo regret
        pull_history = np.array(self.history)[:, 0]
        Rn = 0
        for a, delta_a in enumerate(self.suboptimality_gaps):
            T_a = len(pull_history[(pull_history==a)])
            Rn += T_a * delta_a
        print(f"Regret at round {len(pull_history)}: {Rn}")
        return Rn


class BernoulliBandit(FiniteStochasticBandit):
    def __init__(self, means):
        FiniteStochasticBandit.__init__(self, [bernoulli(mean) for mean in means])