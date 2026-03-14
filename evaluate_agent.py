from stable_baselines3 import PPO
from rl_environment import EEGEnv
import numpy as np

def evaluate(features, prototypes):

    model = PPO.load("eeg_rl_agent")

    env = EEGEnv(features, prototypes)

    state, _ = env.reset()

    selected = []

    done = False

    while not done:

        action, _ = model.predict(state)

        if action == 1:
            selected.append(env.index)

        state, reward, done, _, _ = env.step(action)

    print("Selected segments:", selected)
    print("Total selected:", len(selected))