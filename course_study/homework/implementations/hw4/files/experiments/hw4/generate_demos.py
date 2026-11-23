"""Generate seeded SO100 demonstrations for behavioral cloning with the HW2 inverse kinematics solver."""
from pathlib import Path
import sys

import numpy as np

import torch
import mujoco

OUT = Path.cwd()
HW = Path(__file__).resolve().parents[2] / "hw4_reinforcement_learning"
sys.path.insert(0, str(HW))
from envs.so100_rl_env import SO100RLEnv

class PositionIK:
    """Solve once per goal and hold the same joint command throughout the episode."""
    device = torch.device("cpu")

    def __init__(self, env):
        sys.path.insert(0, str(HW.parent))
        from hw2_robot_control_mdps.exercises.ex1 import ik_track
        self.solve = ik_track
        self.env = env
        self.plan = mujoco.MjData(env.model)

    def predict_action(self, obs):
        if self.env.current_step == 0:
            model = self.env.model
            self.plan.qpos[:] = self.env.data.qpos
            mujoco.mj_forward(model, self.plan)
            q = self.solve(model, self.plan, "ee_site", self.env.data.mocap_pos[0], max_iters=500)
            lo, hi = model.jnt_range.T
            self.action = torch.tensor(np.clip(2*(q-lo)/(hi-lo)-1, -1, 1),
                                       dtype=torch.float32).unsqueeze(0)
        return self.action

def collect(split, seeds):
    env = SO100RLEnv((HW / "assets/mujoco/so100_pos_ctrl.xml").resolve())
    teacher = PositionIK(env)
    states = np.empty((len(seeds), env.max_episode_length, env.state_dim), dtype=np.float32)
    commands = np.empty((len(seeds), env.action_dim), dtype=np.float32)
    errors = np.empty((len(seeds), env.max_episode_length), dtype=np.float32)
    try:
        for i, seed in enumerate(seeds):
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            cmd = teacher.predict_action(None).numpy()[0].copy()
            commands[i] = cmd
            for step in range(env.max_episode_length):
                states[i, step] = obs
                obs, _, terminated, truncated, info = env.step(cmd)
                errors[i, step] = info["ee_tracking_error"]
                assert (terminated or truncated) == (step == env.max_episode_length-1)
            if (i+1) % 512 == 0:
                print(f"{split}: {i+1}/{len(seeds)} episodes", flush=True)
    finally:
        env.close()
    assert np.all(np.isfinite(states)) and np.max(abs(commands)) <= 1
    assert np.all(np.isfinite(errors))
    np.savez_compressed(OUT / f"{split}.npz", states=states, commands=commands,
                        seeds=np.array(seeds), errors_m=errors)
    print(f"{split}: {len(seeds)} episodes, teacher mean error {errors.mean()*100:.3f} cm")


if __name__ == "__main__":
    if (OUT / "demonstrations.npz").exists():
        raise FileExistsError("demonstrations.npz already exists")
    collect("demonstrations", range(202000,204048))
