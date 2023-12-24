import numpy as np

def Greedy(bandit, n):
    """ 
    Greedy algorithm. Choose each action once, and subsequently choose the action with the largest average observed so far.

    Parameters:
        bandit: Bandit instance
        n: length of horizon, n > K
    """

    k = bandit.K()
    if n <= k:
        raise ValueError("Horizon too short for greedy algorithm.")

    mean_rewards = np.zeros(k)

    for t in range(n):
        a = -1
        # choose an arm
        if t < k:
            a = t
        else:
            max_indices = np.where(mean_rewards == mean_rewards.max())[0]
            a = np.random.choice(max_indices) # break ties randomly
        
        # pull the arm
        reward = bandit.pull(a)

        # update the mean vector
        history = np.array(bandit.history)
        mean_rewards[a] = np.mean(history[:, 1][(history[:, 0]==a)])
