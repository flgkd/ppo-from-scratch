import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

from model import Actor, Critic


class PPOAgent:

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()

        self.whether_use_GPU = False

        self.initial_actor_lr = 3e-4
        self.initial_critic_lr = 1e-3

        # use GPU
        self.device = torch.device(
            "cuda" if (self.whether_use_GPU and torch.cuda.is_available())
            else "cpu"
        )

        print("Using device:", self.device)

        self.actor.to(self.device)
        self.critic.to(self.device)

        # specify optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.initial_critic_lr)

        # set clip parameter
        self.clip_epsilon = 0.2

        # set epoch num
        self.ppo_epochs = 4

        # set KL early stopping threshold
        self.target_kl = 0.01

    def select_action(self, state):
        # change the state to the tensor and batch
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # actor output the action probabilities
        probs = self.actor(state_tensor)

        # create the Categorical distribution
        dist = Categorical(probs)

        # choose an action
        action = dist.sample()

        # record the selected action's log probability
        log_prob = dist.log_prob(action).detach()

        # record the state value
        value = self.critic(state_tensor)

        return action.item(), log_prob.item(), value.item()

    def update(self, memory):

        states = torch.FloatTensor(np.array(memory.states)).to(self.device)

        actions = torch.LongTensor(memory.actions).to(self.device)

        old_log_probs = torch.FloatTensor(memory.log_probs).to(self.device)

        # Monte Carlo
        # returns = memory.compute_returns()

        # advantages = memory.compute_advantages(returns)

        # GAE
        advantages, returns = memory.compute_gae()

        advantages = advantages.to(self.device)

        returns = returns.to(self.device)

        # normalize advantage (important)
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

        # use mini-batches
        dataset_size = states.size(0)

        mini_batch_size = min(64, dataset_size)  # can adjust

        indices = np.arange(dataset_size)

        for epoch in range(self.ppo_epochs):

            kl_list = []

            np.random.shuffle(indices)

            for start in range(0, dataset_size, mini_batch_size):

                end = start + mini_batch_size

                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # ====== forward ======

                # compute new log probs
                # actor output the new action probabilities
                probs = self.actor(batch_states)

                # create the Categorical distribution
                dist = Categorical(probs)

                # record all new actions' log probabilities
                new_log_probs = dist.log_prob(batch_actions)

                # compute approximation KL
                approx_kl = (batch_old_log_probs - new_log_probs).mean()

                kl_list.append(approx_kl.detach())

                # ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # clip
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                # actor loss
                # apply entropy bonus
                entropy = dist.entropy().mean()

                entropy_coef = 0.001  # 0.0005 or 0.001

                actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()\
                             - entropy_coef * entropy

                # critic loss
                values = self.critic(batch_states).view(-1)

                critic_loss = F.mse_loss(values, batch_returns)

                # ====== backward ======

                # update actor and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                actor_loss.backward()
                critic_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # ===== KL early stopping check =====
                mean_kl = torch.stack(kl_list).mean()
                if mean_kl > self.target_kl:
                    print(f"Early stopping at epoch {epoch}, KL={mean_kl.item():.6f}")
                    break

        return actor_loss.item(), critic_loss.item()

    # use learning rate decay
    def decay_learning_rate(self, current_episode, max_episode):

        frac = 1 - (current_episode / max_episode)

        frac = max(frac, 0.1)

        new_actor_lr = self.initial_actor_lr * frac

        new_critic_lr = self.initial_critic_lr * frac

        for param_group in self.actor_optimizer.param_groups:

            param_group['lr'] = new_actor_lr

        for param_group in self.critic_optimizer.param_groups:

            param_group['lr'] = new_critic_lr


















