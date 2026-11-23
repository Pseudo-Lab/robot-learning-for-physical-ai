from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import pytest

from conftest import PROJECT_ROOT, add_homework_to_path, requested_device


add_homework_to_path("hw4_reinforcement_learning")
mdp = importlib.import_module("hw4_reinforcement_learning.exercises.ex1_mdp")
dqn_module = importlib.import_module("hw4_reinforcement_learning.exercises.ex2_dqn")
ppo_module = importlib.import_module("hw4_reinforcement_learning.exercises.ex3_ppo")
sac_module = importlib.import_module("hw4_reinforcement_learning.exercises.ex4_sac")
buffers = importlib.import_module("rl.buffers")
environment_module = importlib.import_module("envs.so100_rl_env")


def make_robot_env():
    return environment_module.SO100RLEnv(
        xml_path=PROJECT_ROOT
        / "hw4_reinforcement_learning/assets/mujoco/so100_pos_ctrl.xml",
        render_mode=None,
    )


class TinyMDP:
    n_states = 2
    n_actions = 2
    P = {
        0: {
            0: [(1.0, 0, 0.0, False)],
            1: [(1.0, 1, 1.0, True)],
        },
        1: {
            0: [(1.0, 1, 0.0, True)],
            1: [(1.0, 1, 0.0, True)],
        },
    }


def test_policy_and_value_iteration_match() -> None:
    pi = mdp.PolicyIteration(TinyMDP(), theta=1e-10, gamma=0.9)
    vi = mdp.ValueIteration(TinyMDP(), theta=1e-10, gamma=0.9)
    pi_v, pi_policy = pi.policy_iteration()
    vi_v, vi_policy = vi.value_iteration()
    np.testing.assert_allclose(pi_v, vi_v, atol=1e-8)
    assert np.argmax(pi_policy[0]) == 1
    assert np.argmax(vi_policy[0]) == 1


def test_dqn_replay_target_update_and_checkpoint(tmp_path: Path) -> None:
    device = requested_device()
    env = gym.make("CartPole-v1")
    replay = dqn_module.ReplayBuffer(20)
    agent = dqn_module.DQN(4, 16, 2, 1e-3, 0.9, 0.1, 1, device)
    state, _ = env.reset(seed=11)
    for _ in range(12):
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(state, action, reward, next_state, done)
        state = env.reset()[0] if done else next_state
    arrays = replay.sample(8)
    assert arrays[0].shape == (8, 4)
    before = [p.detach().clone() for p in agent.q_net.parameters()]
    agent.update(
        dict(zip(("states", "actions", "rewards", "next_states", "dones"), arrays))
    )
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.q_net.parameters()))
    for online, target in zip(
        agent.q_net.parameters(), agent.target_q_net.parameters()
    ):
        torch.testing.assert_close(online, target)
    checkpoint = tmp_path / "dqn.pt"
    agent.save(checkpoint)
    clone = dqn_module.DQN(4, 16, 2, 1e-3, 0.9, 0.0, 1, device)
    clone.load(checkpoint)
    assert clone.predict_action(np.zeros(4, dtype=np.float32)) in (0, 1)
    env.close()

    target_agent = dqn_module.DQN(2, 4, 2, 0.0, 0.9, 0.0, 100, device)
    with torch.no_grad():
        for parameter in target_agent.q_net.parameters():
            parameter.zero_()
        for parameter in target_agent.target_q_net.parameters():
            parameter.zero_()
        target_agent.target_q_net.fc2.bias.copy_(
            torch.tensor([0.0, 2.0], device=device)
        )
    target_loss = target_agent.update(
        {
            "states": np.zeros((2, 2), dtype=np.float32),
            "actions": [0, 0],
            "rewards": [1.0, 2.0],
            "next_states": np.zeros((2, 2), dtype=np.float32),
            "dones": [False, True],
        }
    )
    expected_targets = torch.tensor([2.8, 2.0])
    expected_loss = expected_targets.square().mean().item()
    assert target_loss == pytest.approx(expected_loss, rel=1e-6)


def test_ppo_clipping_kl_update_and_checkpoint(tmp_path: Path) -> None:
    device = requested_device()
    env = make_robot_env()
    size, obs_dim, act_dim = 8, env.state_dim, env.action_dim
    agent = ppo_module.PPOAgent(
        obs_dim,
        act_dim,
        [16],
        n_steps=size,
        mini_batch_size=4,
        n_epochs=2,
        entropy_coeff=0.01,
        device=device,
    )
    observations, actions, logps, means, stds, values, rewards = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    obs_array, _ = env.reset(seed=5)
    with torch.no_grad():
        for _ in range(size):
            obs_tensor = torch.as_tensor(
                obs_array, dtype=torch.float32, device=device
            ).unsqueeze(0)
            action, action_env, value, logp, mu, std = agent.select_action(obs_tensor)
            next_obs, reward, terminated, truncated, _ = env.step(
                action_env.cpu().numpy().squeeze(0)
            )
            observations.append(obs_tensor.squeeze(0))
            actions.append(action.squeeze(0))
            logps.append(logp)
            means.append(mu.squeeze(0).clone())
            stds.append(std.squeeze(0).clone())
            values.append(value)
            rewards.append(reward)
            obs_array = env.reset()[0] if terminated or truncated else next_obs
    obs = torch.stack(observations)
    act = torch.stack(actions)
    logp = torch.tensor(logps, dtype=torch.float32, device=device)
    mu = torch.stack(means)
    std = torch.stack(stds)
    val = torch.tensor(values, dtype=torch.float32, device=device)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    batch = buffers.RolloutBatch(
        obs=obs,
        act=act,
        logp=logp,
        mu=mu,
        std=std,
        val=val,
        ret=reward_tensor + 0.99 * val.detach(),
        adv=(reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8),
    )
    zero_kl = agent.compute_kl_mean(mu, std, mu, std)
    torch.testing.assert_close(zero_kl, torch.zeros_like(zero_kl))
    clipped = agent.compute_surrogate_loss(
        torch.full((4,), np.log(2.0), device=device),
        torch.zeros(4, device=device),
        torch.ones(4, device=device),
    )
    assert clipped.item() == pytest.approx(-1.2, rel=1e-6)
    stats = agent.update(batch)
    assert np.isfinite(stats.mean_kl)
    checkpoint = tmp_path / "ppo.pt"
    agent.save(checkpoint)
    agent.load(checkpoint)
    env.close()


def test_sac_losses_soft_update_and_checkpoint(tmp_path: Path) -> None:
    device = requested_device()
    env = make_robot_env()
    obs_dim, act_dim, batch_size = env.state_dim, env.action_dim, 16
    agent = sac_module.SACAgent(
        obs_dim,
        act_dim,
        [32, 32],
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        gamma=0.99,
        tau=0.05,
        init_alpha=0.2,
        target_entropy=None,
        device=device,
    )
    replay = buffers.ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, max_size=32, device=device
    )
    obs_array, _ = env.reset(seed=7)
    for _ in range(batch_size):
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
        action = agent.sample_action(obs_tensor.unsqueeze(0)).squeeze(0)
        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        replay.store(
            obs_tensor,
            action,
            reward,
            torch.as_tensor(next_obs, dtype=torch.float32, device=device),
            terminated or truncated,
        )
        obs_array = env.reset()[0] if terminated or truncated else next_obs
    batch = replay.sample_batch(batch_size)
    target_before = [p.detach().clone() for p in agent.critic_target.parameters()]
    stats = agent.update(batch)
    assert all(np.isfinite(value) for value in vars(stats).values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, agent.critic_target.parameters())
    )
    action = agent.sample_action(torch.zeros(3, obs_dim, device=device))
    assert action.shape == (3, act_dim)
    assert torch.all(action.abs() <= 1.0)
    checkpoint = tmp_path / "sac.pt"
    agent.save(checkpoint)
    agent.load(checkpoint)
    env.close()


def test_sac_time_limit_keeps_bootstrapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = importlib.import_module("hw4_reinforcement_learning.scripts.train_sac")
    use_cuda = requested_device().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: use_cuda)
    (tmp_path / "assets").symlink_to(
        PROJECT_ROOT / "hw4_reinforcement_learning/assets", target_is_directory=True
    )
    monkeypatch.setattr(trainer, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        trainer,
        "SAC_PARAMETERS",
        {
            **trainer.SAC_PARAMETERS,
            "learning_start_steps": 30,
            "train_freq": 30,
            "total_iterations": 1,
            "gradient_steps": 1,
            "replay_size": 32,
            "batch_size": 4,
        },
    )
    seen = []
    original_store = trainer.ReplayBuffer.store

    def observe_store(buffer, **transition):
        seen.append(transition["done"])
        original_store(buffer, **transition)

    monkeypatch.setattr(trainer.ReplayBuffer, "store", observe_store)
    trainer.main()
    # Step 30 is the actual environment's time limit.
    assert seen == [False] * 30


def test_greedy_planners_distinguish_close_rewards_and_keep_exact_ties() -> None:
    class OneStepMDP:
        n_states = 1
        n_actions = 3
        P = [[[(1.0, 0, reward, True)] for reward in (10.0, 10.00001, 10.00001)]]

    env = OneStepMDP()
    for solve in (
        mdp.PolicyIteration(env, theta=1e-10).policy_iteration,
        mdp.ValueIteration(env, theta=1e-10).value_iteration,
    ):
        _, policy = solve()
        np.testing.assert_array_equal(policy[0], [0.0, 0.5, 0.5])
