import torch

import gymnasium as gym
import numpy as np

from ppo.agent import PPOAgent
from ppo.memory import Memory

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1")

agent = PPOAgent()
memory = Memory()

max_episodes = 500

# ===== use rollout update =====
rollout_steps = 512

reward_history = []
actor_loss_history = []
critic_loss_history = []

# start the episode (Here _ is the info)
state, _ = env.reset()

episode_reward = 0
episode = 0
global_step = 0

while episode < max_episodes:

    action, log_prob, value = agent.select_action(state)

    # generate the new state and reward (Here _ is the info)
    # note that Gymnasium returns five values here
    next_state, reward, terminated, truncated, _ = env.step(action)

    # Terminated: The task has truly ended.
    # (e.g., in CartPole, the pole has fallen over, or the cart has moved out of bounds).
    # Truncated: The task was forcibly stopped.
    # (e.g., you set a maximum limit of 500 steps,
    # and the system forced the episode to end even though the pole hadn't fallen yet).
    done = terminated or truncated

    memory.store(
        state,
        action,
        log_prob,
        reward,
        done,
        value
    )

    episode_reward += reward

    state = next_state

    global_step += 1

    # ===== If episode ends =====
    if done:

        # save episode reward
        reward_history.append(episode_reward)

        print(
            f"Episode {episode:4d} | "
            f"Reward {episode_reward:4.0f}"
        )

        episode_reward = 0
        state, _ = env.reset()
        episode += 1

        # apply learning rate decay per episode
        agent.decay_learning_rate(episode, max_episodes)

    # ===== Fixed rollout update =====
    if len(memory.states) >= rollout_steps:

        # update ppo
        actor_loss, critic_loss = agent.update(memory)

        # save loss
        actor_loss_history.append(actor_loss)
        critic_loss_history.append(critic_loss)

        print(f"Update | Actor Loss {actor_loss:.4f} | Critic Loss {critic_loss:.4f}")

        # clear memory
        memory.clear()


# ================== Visualization ==================
def plot_ppo_results(reward_hist, actor_loss_hist, critic_loss_hist, window=20):
    """
    Generate an integrated dashboard for ppo training analysis.

    Args:
        reward_hist: List of total rewards per episode.
        actor_loss_hist: List of actor (policy) losses per update.
        critic_loss_hist: List of critic (value) losses per update.
        window: Sliding window size for moving average.
    """
    # Use a modern visual style
    plt.style.use('seaborn-v0_8-muted')

    # Create a layout with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"ppo Training Analysis (MA Window: {window})", fontsize=16)

    # Helper function to plot data with its corresponding x-axis label
    def draw_subplot(ax, data, title, ylabel, xlabel, color):
        # 1. Plot raw data (faded)
        ax.plot(data, alpha=0.25, color=color, label='Raw Data')

        # 2. Plot moving average if enough data points exist
        if len(data) >= window:
            smooth_data = np.convolve(data, np.ones(window) / window, mode='valid')
            # Align moving average to the end of the window
            x_axis = np.arange(window - 1, len(data))
            ax.plot(x_axis, smooth_data, color=color, linewidth=2, label='Moving Avg')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left', fontsize='small')

    # Rewards are logged per Episode
    draw_subplot(axs[0], reward_hist, "Episode Reward", "Reward", "Episode", "green")

    # Losses are logged per Update Step (Rollout completion)
    draw_subplot(axs[1], actor_loss_hist, "Actor Loss", "Loss", "Update Step", "blue")
    draw_subplot(axs[2], critic_loss_hist, "Critic Loss", "MSE Loss", "Update Step", "red")

    # Prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Run visualization
plot_ppo_results(reward_history, actor_loss_history, critic_loss_history, window=25)


# ================== Save trained model ==================
torch.save(agent.actor.state_dict(), "ppo_cartpole_actor.pth")
torch.save(agent.critic.state_dict(), "ppo_cartpole_critic.pth")

print("Model saved successfully.")













