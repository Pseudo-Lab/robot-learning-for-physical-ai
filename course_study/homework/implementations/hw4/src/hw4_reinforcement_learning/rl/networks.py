from ethz_starter.hw4_reinforcement_learning.rl import networks as course
from ethz_starter.hw4_reinforcement_learning.rl.networks import (
    DoubleQNet,
    QNet,
    ValueNet,
    build_mlp,
)


def _action_api(self):
    raise RuntimeError("Use act(), act_inference(), or update_distribution()")


class GaussianActor(course.GaussianActor):
    forward = _action_api


class SquashedGaussianActor(course.SquashedGaussianActor):
    forward = _action_api
