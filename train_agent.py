from sb3_contrib import RecurrentPPO
from rl_environment import EEGEnv


def train(features, prototypes):
    env = EEGEnv(features, prototypes)
    policy_kwargs = dict(
        lstm_hidden_size=128,
        net_arch=[64, 32]
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"
    )
    model.learn(total_timesteps=100000)
    model.save("eeg_rl_agent")
    print("Model saved!")