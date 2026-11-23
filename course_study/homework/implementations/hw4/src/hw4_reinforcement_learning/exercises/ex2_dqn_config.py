from ethz_starter.hw4_reinforcement_learning.exercises.ex2_dqn_config import (
    DQN_PARAMETERS as defaults,
)

DQN_PARAMETERS = dict(defaults, lr=1e-3, epsilon=0.05, target_update=20, hidden_dim=128)
