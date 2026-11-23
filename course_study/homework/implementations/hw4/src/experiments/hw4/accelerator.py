"""Collect rollouts with a CPU copy of the policy and train on CUDA minibatches."""
from copy import copy, deepcopy
from dataclasses import fields
from inspect import getsource
from textwrap import dedent

import torch


def host_policy(agent, *, value):
    acting = copy(agent)
    acting.device = torch.device("cpu")
    acting.actor = deepcopy(agent.actor).cpu()
    if value:
        acting.critic = deepcopy(agent.critic).cpu()
    return acting


def refresh(acting, agent, *, value=False):
    acting.actor.load_state_dict(agent.actor.state_dict())
    if value:
        acting.critic.load_state_dict(agent.critic.state_dict())


def transfer(batch):
    return type(batch)(**{f.name: getattr(batch, f.name).to("cuda") for f in fields(batch)})


class SACUpdates:
    """Run SAC updates on CUDA and copy the four averaged statistics to the host once per burst."""

    def __init__(self, agent, stats_cls):
        self.agent = agent
        self.stats_cls = stats_cls
        native = type(agent).update
        source = dedent(getsource(native))
        assert source.count(".item()") == 4
        scope = dict(native.__globals__)
        exec(source.replace(".item()", ".detach()"), scope)
        self.update = scope["update"]
        for opt in (agent.actor_optimizer, agent.critic_optimizer, agent.alpha_optimizer):
            for group in opt.param_groups:
                group.update(capturable=True, fused=True)
        self.count = 0
        self.graph = None

    def one(self, batch):
        if self.count < 3:
            stats = self.update(self.agent, batch)
        else:
            if self.graph is None:
                self.fixed = type(batch)(**{f.name: getattr(batch, f.name).clone() for f in fields(batch)})
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.stats = self.update(self.agent, self.fixed)
            for field in fields(batch):
                getattr(self.fixed, field.name).copy_(getattr(batch, field.name))
            self.graph.replay()
            stats = self.stats
        self.count += 1
        return torch.stack([getattr(stats, f.name) for f in fields(stats)])

    def burst(self, replay, batch_size, gradient_steps):
        samples = transfer(replay.sample_batch(batch_size * gradient_steps))
        batches = {f.name: getattr(samples, f.name).view(gradient_steps, batch_size, *getattr(samples, f.name).shape[1:])
                   for f in fields(samples)}
        stats = []
        for index in range(gradient_steps):
            batch = type(samples)(**{name: value[index] for name, value in batches.items()})
            stats.append(self.one(batch))
        values = torch.stack(stats).mean(0).tolist()
        if not all(__import__("math").isfinite(v) for v in values):
            raise ArithmeticError(f"Nonfinite SAC statistics after {self.count} updates")
        return self.stats_cls(*values)


def prepare(source, method):
    marker = "    buffer = RolloutBuffer(" if method == "ppo" else "    replay_buffer = ReplayBuffer("
    assert source.count(marker) == 1
    setup = ["    acting = host_policy(agent, value=" + str(method == "ppo") + ")", '    device = torch.device("cpu")']
    if method == "sac":
        setup.append("    updates = SACUpdates(agent, SACUpdateStats)")
    source = source.replace(marker, "\n".join(setup) + "\n" + marker)
    if method == "ppo":
        source = source.replace("agent.select_action(obs)", "acting.select_action(obs)").replace("agent.critic(", "acting.critic(")
        old = "stats = agent.update(rollout_batch)"
        assert source.count(old) == 1
        source = source.replace(old, "stats = agent.update(transfer(rollout_batch))\n        refresh(acting, agent, value=True)")
    else:
        start = source.index("            mean_stats = SACUpdateStats.init_lists()")
        end = source.index("            it += 1", start)
        source = source[:start] + "            mean_stats = updates.burst(replay_buffer, batch_size, gradient_steps)\n            refresh(acting, agent)\n" + source[end:]
        source = source.replace("agent.sample_action(obs)", "acting.sample_action(obs)")
        source = source.replace("    final_model_path =", "    assert updates.count == it * gradient_steps\n    assert int(agent.alpha_optimizer.state[agent.log_alpha]['step'].item()) == updates.count\n    final_model_path =")
    return source.replace("eval_env, agent, num_episodes", "eval_env, acting, num_episodes")
