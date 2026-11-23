import argparse
import time
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from hw2_robot_control_mdps.env.so100_tracking_env import SO100TrackEnv
from hw2_robot_control_mdps.exercises.ex1 import build_keypoints
from hw2_robot_control_mdps.scripts.utils import refresh_markers

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "logs/so100_tracking"
XML_PATH = ROOT / "so101_gym/assets/so100_pos_ctrl.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on SO100 tracking")
    parser.add_argument("--load_run", type=str, default="1")
    parser.add_argument("--checkpoint", type=str, default="500")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    policy_path = args.model_path or (
        EXP_DIR / f"so100_tracking_{args.load_run}" / f"model_{args.checkpoint}.zip"
    )
    policy = PPO.load(policy_path, device=args.device)
    env = SO100TrackEnv(XML_PATH, render_mode=None if args.headless else "human")
    errors = []
    try:
        for episode in range(10):
            obs, _ = env.reset(seed=args.seed + episode)
            for _ in range(int(2 / env.ctrl_timestep)):
                tick = time.monotonic()
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if not args.headless:
                    if not env.viewer.is_running():
                        break
                    time.sleep(max(0, env.ctrl_timestep - (time.monotonic() - tick)))
                if terminated or truncated:
                    break
            errors.append(info["ee_tracking_error"])
            print(f"Final EE tracking error: {errors[-1]:.4f}")
            if not args.headless and not env.viewer.is_running():
                break
    finally:
        env.close()
    if errors:
        print(f"Average final EE tracking error: {np.mean(errors):.4f}")
