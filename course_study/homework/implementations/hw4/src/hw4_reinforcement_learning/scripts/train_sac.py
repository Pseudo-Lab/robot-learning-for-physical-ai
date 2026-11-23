from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw4_reinforcement_learning.envs.so100_rl_env import SO100RLEnv
from hw4_reinforcement_learning.exercises.ex4_sac import SACAgent, SACUpdateStats
from hw4_reinforcement_learning.exercises.ex4_sac_config import SAC_PARAMETERS
from hw4_reinforcement_learning.rl.buffers import ReplayBuffer
from hw4_reinforcement_learning.rl.common import set_seed
from ethz_starter.hw4_reinforcement_learning.scripts.train_sac import evaluate_policy

ROOT_DIR = Path(__file__).resolve().parents[1]


def main():
    cfg = SAC_PARAMETERS
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xml = ROOT_DIR / "assets/mujoco/so100_pos_ctrl.xml"
    env, monitor = SO100RLEnv(xml_path=xml), SO100RLEnv(xml_path=xml)
    folder = ROOT_DIR / "logs/sac" / datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
    writer = SummaryWriter(log_dir=str(folder))
    try:
        agent = SACAgent(
            obs_dim=env.state_dim,
            act_dim=env.action_dim,
            device=device,
            **{
                k: cfg[k]
                for k in (
                    "hidden_sizes",
                    "actor_lr",
                    "critic_lr",
                    "alpha_lr",
                    "gamma",
                    "tau",
                    "init_alpha",
                    "target_entropy",
                )
            },
        )
        if cfg.get("resume_from"):
            agent.load(cfg["resume_from"])
        replay = ReplayBuffer(
            obs_dim=env.state_dim,
            act_dim=env.action_dim,
            max_size=cfg["replay_size"],
            device=device,
        )
        state, _ = env.reset()
        iteration = step = evaluations = 0
        stats = None
        while iteration < cfg["total_iterations"]:
            step += 1
            agent.train_mode()
            with torch.inference_mode():
                obs = torch.as_tensor(state, dtype=torch.float32, device=device)
                if step < cfg["learning_start_steps"] and not cfg.get("resume_from"):
                    action = torch.empty(env.action_dim, device=device).uniform_(-1, 1)
                else:
                    action = agent.sample_action(obs.unsqueeze(0))[0]
                nxt, reward, terminal, timeout, _ = env.step(action.cpu().numpy())
                replay.store(
                    obs=obs,
                    act=action,
                    rew=reward,
                    next_obs=torch.as_tensor(nxt, device=device, dtype=obs.dtype),
                    done=terminal,
                )
                state = env.reset()[0] if terminal or timeout else nxt
            if step >= cfg["learning_start_steps"] and step % cfg["train_freq"] == 0:
                rows = SACUpdateStats.init_lists()
                for _ in range(cfg["gradient_steps"]):
                    rows.append(agent.update(replay.sample_batch(cfg["batch_size"])))
                stats = rows.mean()
                iteration += 1
            if step % cfg["eval_freq"] == 0:
                evaluations += 1
                returns, length, error = evaluate_policy(monitor, agent)
                for name, value in zip(
                    ("return", "length", "ee_tracking_error"), (returns, length, error)
                ):
                    writer.add_scalar("eval/" + name, value, evaluations)
                if stats:
                    for name, value in vars(stats).items():
                        writer.add_scalar("train/" + name, value, evaluations)
                writer.add_scalar("train/step", step, evaluations)
                if evaluations % cfg["save_interval"] == 0:
                    agent.save(folder / f"iter_{iteration}.pt")
        agent.save(folder / f"iter_{iteration}.pt")
    finally:
        env.close()
        monitor.close()
        writer.close()


if __name__ == "__main__":
    main()
