from itertools import chain
import torch
from torch import nn
from ethz_starter.hw4_reinforcement_learning.exercises.ex3_ppo import (
    PPOAgent as CoursePPO,
    PPOUpdateStats,
)


class PPOAgent(CoursePPO):
    def select_action(self, obs: torch.Tensor):
        with torch.inference_mode():
            action = self.actor.act(obs)
            action_clipped = torch.clamp(action, -1.0, 1.0)
            action_log_prob = self.actor.get_actions_log_prob(action).item()
            action_mu = self.actor.action_mean
            action_std = self.actor.action_std
            value = self.critic(obs).item()
        return (action, action_clipped, value, action_log_prob, action_mu, action_std)

    def compute_kl_mean(self, old_mu_batch, old_std_batch, mu_batch, std_batch):
        kl_per_dim = (
            torch.log(std_batch / old_std_batch)
            + (old_std_batch.pow(2) + (old_mu_batch - mu_batch).pow(2))
            / (2.0 * std_batch.pow(2))
            - 0.5
        )
        kl_per_sample = kl_per_dim.sum(dim=-1)
        return kl_per_sample.mean()

    def compute_surrogate_loss(self, logp_batch, old_logp_batch, adv_batch):
        ratio = torch.exp(logp_batch - old_logp_batch)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        surrogate_loss = -torch.minimum(
            ratio * adv_batch, clipped_ratio * adv_batch
        ).mean()
        return self.surrogate_loss_coeff * surrogate_loss

    def compute_value_loss(self, val_batch, old_val_batch, ret_batch):
        value_loss_unclipped = (val_batch - ret_batch).pow(2)
        value_clipped = old_val_batch + torch.clamp(
            val_batch - old_val_batch, -self.clip_ratio, self.clip_ratio
        )
        value_loss_clipped = (value_clipped - ret_batch).pow(2)
        value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped).mean()
        return self.value_loss_coeff * value_loss

    def compute_entropy_loss(self, entropy_batch):
        return -self.entropy_coeff * entropy_batch.mean()

    def update(self, rollout_batch):
        updates = []
        for batch in self.mini_batch_generator(rollout_batch):
            self.actor.update_distribution(batch.obs)
            logp = self.actor.get_actions_log_prob(batch.act)
            kl = self.compute_kl_mean(
                batch.mu, batch.std, self.actor.action_mean, self.actor.action_std
            ).item()
            self.learning_rate = self.adjust_learning_rate(kl, self.learning_rate)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
            policy_loss = self.compute_surrogate_loss(logp, batch.logp, batch.adv)
            value_loss = self.compute_value_loss(
                self.critic(batch.obs), batch.val, batch.ret
            )
            entropy = self.actor.entropy
            loss = policy_loss + value_loss + self.compute_entropy_loss(entropy)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                chain(self.actor.parameters(), self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()
            updates.append(
                (kl, policy_loss.item(), value_loss.item(), entropy.mean().item())
            )
        return PPOUpdateStats(*(sum(column) / len(column) for column in zip(*updates)))
