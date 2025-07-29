#from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from franka_env import FrankaReachEnv
from matplotlib import pyplot as plt

env = FrankaReachEnv(render=False)

model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_franka_tensorboard/", device="cpu")
model.learn(total_timesteps=100_000)


obs, _ = env.reset()
rewards = []
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    rewards.append(reward)
    if done:
        break

env.close()
print(rewards)

# Plot the rewards
plt.plot(rewards)
plt.title("Rewards over time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.show()
