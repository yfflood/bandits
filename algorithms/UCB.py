import numpy as np


def UCB(bandit, n, horizon=False):
    """ Upper confidence bound algorithm """
    delta = 0
    k = bandit.K()
    if n <= k:
        raise Warning("Horizon too short for UCB algorithm.")

    UpperConfidenceBound = np.ones(k) * np.inf

    for t in range(n):
        # choose arm
        max_indices = np.where(UpperConfidenceBound==UpperConfidenceBound.max())[0]
        a = np.random.choice(max_indices)
        bandit.pull(a)

        # set confidence level
        if horizon:
            delta = 1 / (n ** 2)
        else:
            delta = 1 / (1 + (t+1) * (np.log(t+1)) ** 2)

        # calculated UCB
        history = np.array(bandit.history)
        mean_rewards = np.mean(history[:, 1][(history[:, 0]==a)])
        T_a = len(history[:, 1][(history[:, 0]==a)])
        bound_width = np.sqrt(2 * np.log(1 / delta) / T_a) 

        UpperConfidenceBound[a] = mean_rewards + bound_width