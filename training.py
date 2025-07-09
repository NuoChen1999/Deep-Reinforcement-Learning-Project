from stable_baselines3 import PPO
from franka_env import FrankaReachEnv

env = FrankaReachEnv(render=False)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_franka_tensorboard/")
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
