import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from hw2_robot_control_mdps.env.so100_tracking_env import SO100TrackEnv
from hw2_robot_control_mdps.scripts.utils import (
    EpisodeLoggingCallback,
    UpdateCheckpointCallback,
    KLAdaptiveLRCallback,
)

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "so101_gym/assets/so100_pos_ctrl.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on SO100 tracking")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments; set 1 for single process",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=500,
        help="Number of PPO update iterations",
    )
    parser.add_argument(
        "--save_checkpt_freq",
        type=int,
        default=50,
        help="Checkpoint every N update iterations",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (cpu or cuda)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the viewer for single-environment training",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_env():
    return SO100TrackEnv(XML)


def main():
    args = parse_args()
    if args.num_envs > 1:
        env = VecMonitor(
            SubprocVecEnv([make_env] * args.num_envs),
            info_keywords=("ee_tracking_error",),
        )
    else:
        env = SO100TrackEnv(XML, render_mode=None if args.headless else "human")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            gamma=0.99,
            ent_coef=0.001,
            vf_coef=1.0,
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=str(ROOT / "logs/so100_tracking"),
        )
        saver = UpdateCheckpointCallback(
            save_path=None,
            name_prefix="model",
            save_freq_updates=args.save_checkpt_freq,
            verbose=0,
        )
        schedule = KLAdaptiveLRCallback(
            target_kl=0.05, init_lr=0.001, min_lr=0.00001, max_lr=0.001
        )
        model.learn(
            args.max_iterations * model.n_steps * model.n_envs,
            tb_log_name="so100_tracking",
            callback=[EpisodeLoggingCallback(), saver, schedule],
        )
        model.save(Path(model.logger.get_dir()) / f"model_{saver.update_counter}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
