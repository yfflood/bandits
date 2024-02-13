import sys
sys.path.append("E:/Research/bandits")


from envs.FiniteStochasticBandit import *
from algorithms.greedy import Greedy
from algorithms.UCB import UCB

from utils.visual import *

import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from tqdm import tqdm


# Experiments on the failure of greedy algorithm
def experiment_4p11(seed=42):
    np.random.seed(seed)

    # Experiment: Tor20-4.11
    n_runs = 1000
    n_rounds = 100
    bandit = BernoulliBandit([0.5, 0.6])
    regrets = []
    for run_id in tqdm(range(n_runs)):
        bandit.restart()
        Greedy(bandit, n=n_rounds)
        regrets.append(bandit.regret())
    
    plt.figure(figsize=(5,5))
    plt.hist(regrets)
    plt.xticks([0,2,4,6,8,10])
    plt.xlabel("Regret")
    plt.ylabel("Frequency")
    plt.show()


def experiment_4p12(seed=42):
    np.random.seed(seed)

    n_runs = 1000
    bandit = BernoulliBandit([0.5, 0.6])
    mean_regrets = []
    se_regrets = []
    for n_rounds in 100*(1+np.arange(10)):
        regrets = []
        for run_id in tqdm(range(n_runs)):
            bandit.restart()
            Greedy(bandit, n=n_rounds)
            regrets.append(bandit.regret())
        mean_regrets.append(np.mean(regrets))
        se_regrets.append(np.std(regrets)/np.sqrt(n_runs))
    
    plt.figure(figsize=(5,5))
    plt.plot(100*np.arange(10), mean_regrets)
    plt.errorbar(100*np.arange(10), mean_regrets, yerr=se_regrets)
    #plt.xticks([0,2,4,6,8,10])
    plt.xlabel("n")
    plt.ylabel("Expected regret")
    plt.show()


if __name__=="__main__":
    experiment_4p12()