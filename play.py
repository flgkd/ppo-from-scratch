import torch
import gymnasium as gym
import time

from ppo.agent import PPOAgent

# Create environment with rendering enabled
env = gym.make("CartPole-v1", render_mode="human")

# Create agent
agent = PPOAgent()

# Load trained weights
agent.actor.load_state_dict(torch.load("ppo_cartpole_actor.pth"))
# agent.critic.load_state_dict(torch.load("ppo_cartpole_critic.pth"))

agent.actor.eval()
# agent.critic.eval()

num_episodes = 5

for episode in range(num_episodes):

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:

        # Deterministic action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            logits = agent.actor(state_tensor)
            # note here use argmax rather than sample
            action = torch.argmax(logits, dim=1).item()

        state, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.02)
        done = terminated or truncated

        total_reward += reward

    print(f"Test Episode {episode} | Reward: {total_reward}")

env.close()

