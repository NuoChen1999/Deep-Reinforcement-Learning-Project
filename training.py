from stable_baselines3 import PPO, SAC, DDPG
from franka_env import FrankaReachEnv
from matplotlib import pyplot as plt
import numpy as np

env = FrankaReachEnv(render=False)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_franka_tensorboard/", device="cpu")
model.learn(total_timesteps=100000)


obs, _ = env.reset()
rewards = []
ee_positions = []

for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    rewards.append(reward)

    # the gripper position
    ee_pos = obs[7:10]
    ee_positions.append(ee_pos)

    if done:
        break

env.close()
print(rewards)

# Plot the rewards
plt.plot(rewards)
plt.title("Rewards over time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.savefig("rewards_plot_PPO.png")

# plot the end-effector positions
target_pos = np.array([0.5, 0.0, 0.6])
"""
plt.figure(figsize=(14,6))
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
})
"""
plt.figure()
plt.grid(True, linestyle=':', alpha=0.7)
plt.plot(np.array(ee_positions)[:,0], label=['Real x'])
plt.plot(np.array(ee_positions)[:,1], label=['Real y'])
plt.plot(np.array(ee_positions)[:,2], label=['Real z'])
plt.plot(np.full(5000, target_pos[0]), label="Target x", linestyle='--')
plt.plot(np.full(5000, target_pos[1]), label="Target y", linestyle='--')
plt.plot(np.full(5000, target_pos[2]), label="Target z", linestyle='--')
plt.xlabel("Step")
plt.ylabel("ee_positions")
plt.title("ee_positions over time (RL only)")
plt.legend()
plt.savefig("ee_positions_plot_PPO.png")