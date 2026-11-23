import torch
from torch.nn import functional as F
from ethz_starter.hw4_reinforcement_learning.exercises.ex4_sac import (
    SACAgent as CourseSAC,
    SACUpdateStats,
)
from rl.buffers import ReplayBatch


class SACAgent(CourseSAC):
    def sample_action(self, obs: torch.Tensor):
        with torch.no_grad():
            action, _ = self.actor.act(obs)
        return action

    def compute_critic_loss(self, obs, act, rew, next_obs, done) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_logp = self.actor.act(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.minimum(q1_next, q2_next) - self.alpha * next_logp
            target_q = rew + self.gamma * (1.0 - done) * q_next
        q1_pred, q2_pred = self.critic(obs, act)
        critic_loss = 0.5 * (
            F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        )
        return critic_loss

    def compute_actor_loss(self, obs, act_new, logp_new) -> torch.Tensor:
        q1_new, q2_new = self.critic(obs, act_new)
        q_new = torch.minimum(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * logp_new - q_new).mean()
        return actor_loss

    def compute_alpha_loss(self, logp_new) -> torch.Tensor:
        alpha_loss = -(
            self.log_alpha * (logp_new + self.target_entropy).detach()
        ).mean()
        return alpha_loss

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    (1.0 - self.tau) * target_param.data + self.tau * param.data
                )

    def update(self, batch: ReplayBatch) -> SACUpdateStats:
        critic = self.compute_critic_loss(
            batch.obs, batch.act, batch.rew, batch.next_obs, batch.done
        )
        self.critic_optimizer.zero_grad()
        critic.backward()
        self.critic_optimizer.step()
        action, logp = self.actor.act(batch.obs)
        actor = self.compute_actor_loss(batch.obs, action, logp)
        self.actor_optimizer.zero_grad()
        actor.backward()
        self.actor_optimizer.step()
        temperature = self.compute_alpha_loss(logp)
        self.alpha_optimizer.zero_grad()
        temperature.backward()
        self.alpha_optimizer.step()
        self.soft_update_targets()
        return SACUpdateStats(
            actor.item(), critic.item(), temperature.item(), self.alpha.item()
        )
