from datetime import datetime
import inspect
from pathlib import Path
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.so100_rl_env import SO100RLEnv
from exercises.ex3_ppo import PPOAgent
from exercises.ex3_ppo_config import PPO_PARAMETERS
from rl.buffers import RolloutBuffer
from rl.common import set_seed
from ethz_starter.hw4_reinforcement_learning.scripts.train_ppo import evaluate_policy


def main():
    cfg = PPO_PARAMETERS
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xml = ROOT_DIR / "assets/mujoco/so100_pos_ctrl.xml"
    env, monitor = SO100RLEnv(xml), SO100RLEnv(xml)
    folder = ROOT_DIR / "logs/ppo" / datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
    writer = SummaryWriter(str(folder))

    def observe(state):
        return torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    try:
        kwargs = {
            k: v for k, v in cfg.items() if k in inspect.signature(PPOAgent).parameters
        }
        agent = PPOAgent(
            obs_dim=env.state_dim, act_dim=env.action_dim, device=device, **kwargs
        )
        if cfg.get("resume_from"):
            agent.load(cfg["resume_from"])
            agent.learning_rate = agent.optimizer.param_groups[0]["lr"]
        buffer = RolloutBuffer(
            env.state_dim,
            env.action_dim,
            cfg["n_steps"],
            cfg["gamma"],
            cfg["gae_lambda"],
            device,
        )
        obs = observe(env.reset()[0])
        done = False
        for iteration in range(1, cfg["total_iterations"] + 1):
            agent.train_mode()
            with torch.inference_mode():
                for _ in range(cfg["n_steps"]):
                    if done:
                        obs = observe(env.reset()[0])
                    action, command, value, logp, mu, std = agent.select_action(obs)
                    nxt, reward, terminal, timeout, _ = env.step(
                        command[0].cpu().numpy()
                    )
                    nxt = observe(nxt)
                    done = terminal or timeout
                    if timeout:
                        reward += cfg["gamma"] * agent.critic(nxt).item()
                    buffer.store(
                        obs=obs,
                        act=action,
                        rew=reward,
                        done=done,
                        val=value,
                        logp=logp,
                        mu=mu,
                        std=std,
                    )
                    obs = nxt
                buffer.compute_returns(agent.critic(obs).item())
                batch = buffer.get(device=device)
            stats = agent.update(batch)
            returns, length, error = evaluate_policy(monitor, agent)
            for name, value in vars(stats).items():
                writer.add_scalar("train/" + name, value, iteration)
            writer.add_scalar("train/step", iteration * cfg["n_steps"], iteration)
            writer.add_scalar("train/learning_rate", agent.learning_rate, iteration)
            writer.add_scalar(
                "train/global_action_std",
                agent.actor.action_std.mean().item(),
                iteration,
            )
            for name, value in zip(
                ("return", "length", "ee_tracking_error"), (returns, length, error)
            ):
                writer.add_scalar("eval/" + name, value, iteration)
            print(f"PPO iteration {iteration}: mean tracking error {error:.4f} m")
            if (
                iteration % cfg["save_interval"] == 0
                or iteration == cfg["total_iterations"]
            ):
                agent.save(folder / f"iter_{iteration}.pt")
    finally:
        writer.close()
        monitor.close()
        env.close()


if __name__ == "__main__":
    main()
