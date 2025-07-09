import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

class FrankaReachEnv(gym.Env):
    def __init__(self, render=False):
        self.render = render
        if self.render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.timestep = 1/240.

        self.max_steps = 100
        self.step_count = 0

        self.robot_id = None
        self.target_pos = np.array([0.5, 0.0, 0.4])  # x, y, z

        # Action: 7 joint delta positions
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(7,), dtype=np.float32)

        # Observation: 7 joint positions + 3D end-effector position
        obs_low = np.array([-np.pi] * 7 + [-2, -2, 0])
        obs_high = np.array([np.pi] * 7 + [2, 2, 2])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        self.step_count = 0

        plane = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        # Reset joints
        for i in range(7):
            p.resetJointState(self.robot_id, i, 0)

        for _ in range(10):
            p.stepSimulation()

        return self._get_obs(), {}

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        ee_state = p.getLinkState(self.robot_id, 11)[0]
        return np.array(joint_states + list(ee_state), dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        current_q = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        new_q = np.clip(np.array(current_q) + action, -np.pi, np.pi)

        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, new_q[i], force=200)

        for _ in range(10):
            p.stepSimulation()
            if self.render:
                time.sleep(self.timestep)

        obs = self._get_obs()
        ee_pos = obs[-3:]
        dist = np.linalg.norm(ee_pos - self.target_pos)

        reward = -dist
        done = dist < 0.05 or self.step_count >= self.max_steps

        return obs, reward, done, False, {}

    def close(self):
        p.disconnect()
