import gymnasium as gym
from gymnasium import spaces
import numpy as np
from reward import compute_reward


class EEGEnv(gym.Env):

    def __init__(self, features, prototypes):
        super(EEGEnv, self).__init__()
        self.features = features
        self.prototypes = prototypes
        self.index = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(features.shape[1],),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.index = 0
        return self.features[self.index], {}

    def step(self, action):
        feature = self.features[self.index]
        reward = 0

        if action == 1:
            reward = compute_reward(feature, self.prototypes)

        self.index += 1
        done = self.index >= len(self.features)

        if not done:
            next_state = self.features[self.index]
        else:
            next_state = np.zeros_like(feature)

        return next_state, reward, done, False, {}