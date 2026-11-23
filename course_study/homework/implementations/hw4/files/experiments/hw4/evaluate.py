"""Replay the four final GPU-trained policies on the 103 fixed SO100 scenes."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1] / "hw4_reinforcement_learning"
sys.path.insert(0, str(HW))
from envs.so100_rl_env import SO100RLEnv
from rl.networks import GaussianActor, SquashedGaussianActor


@torch.inference_mode()
def evaluate(condition, results):
    settings = json.loads((results / condition / "settings.json").read_text())
    expected = json.loads((results / condition / "episodes.json").read_text())
    if [row["seed"] for row in expected] != list(range(107400, 107503)):
        raise ValueError("Expected the original 100 scenes and the three replacement seeds")
    env = SO100RLEnv(HW / "assets/mujoco/so100_pos_ctrl.xml")
    try:
        cls = GaussianActor if settings["method"] == "ppo" else SquashedGaussianActor
        actor = cls(env.state_dim, env.action_dim, settings["config"]["hidden_sizes"])
        with np.load(HERE / condition / "actor.npz", allow_pickle=False) as state:
            actor.load_state_dict({k: torch.from_numpy(state[k]) for k in state})
        actor.eval()
        errors = {}
        for row in expected:
            seed = row["seed"]
            np.random.seed(seed)
            state, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(state, row["initial"])
            np.testing.assert_array_equal(env.data.mocap_pos[0], row["target"])
            distances = []
            for _ in range(30):
                action = actor.act_inference(torch.as_tensor(state).unsqueeze(0)).clamp(-1, 1)
                state, _, terminated, truncated, info = env.step(action[0].numpy())
                distances.append(info["ee_tracking_error"])
                if terminated or truncated:
                    break
            if len(distances) != 30:
                raise ValueError(f"Unexpected episode length at seed {seed}")
            errors[seed] = float(np.mean(distances))
            np.testing.assert_allclose([errors[seed], distances[-1]],
                                       [row["mean_error_m"], row["final_error_m"]], rtol=0, atol=1e-6)
        original = sum(errors[s] < .05 for s in range(107400, 107500))
        reachable = sum(v < .05 for s, v in errors.items() if s not in (107405, 107410, 107423))
        print(f"{condition}: original {original}/100, reachable {reachable}/100, "
              f"107492 {100*errors[107492]:.3f} cm")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="path to implementations/hw4/results")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for condition in ("rl-ppo", "bc-ppo", "rl-sac", "bc-sac"):
        evaluate(condition, args.results)
