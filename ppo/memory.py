import torch


class Memory:

    def __init__(self):

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    # Monte Carlo
    def compute_returns(self, gamma=0.99):

        returns = []
        G = 0

        # here note the reverse
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):

            if done == 1:

                G = 0

            G = reward + gamma * G

            returns.insert(0, G)

        return torch.FloatTensor(returns)

    def compute_advantages(self, returns):

        values = torch.FloatTensor(self.values)

        advantages = returns - values

        return advantages

    # GAE
    def compute_gae(self, gamma=0.99, lam=0.95):

        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        dones = torch.FloatTensor(self.dones)

        # bootstrap value
        values = torch.cat([values, torch.tensor([0.0])])

        advantages = torch.zeros_like(rewards)

        gae = 0

        for t in reversed(range(len(rewards))):

            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]

            gae = delta + gamma * lam * gae * (1 - dones[t])

            advantages[t] = gae

            returns = advantages + values[:-1]

        return advantages, returns




























