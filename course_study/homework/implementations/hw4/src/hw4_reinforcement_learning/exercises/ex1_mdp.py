import numpy as np


def action_values(env, state, values, gamma):
    return np.array(
        [
            sum(
                prob * (reward + gamma * values[nxt] * (not done))
                for prob, nxt, reward, done in env.P[state][action]
            )
            for action in range(env.n_actions)
        ]
    )


class PolicyIteration:
    def __init__(self, env, theta=1e-3, gamma=0.9):
        self.env, self.theta, self.gamma = env, theta, gamma
        self.v = np.zeros(env.n_states)
        self.pi = np.full((env.n_states, env.n_actions), 1 / env.n_actions)

    def policy_evaluation(self):
        while True:
            new = np.array(
                [
                    self.pi[s] @ action_values(self.env, s, self.v, self.gamma)
                    for s in range(self.env.n_states)
                ]
            )
            delta = np.max(abs(new - self.v))
            self.v = new
            if delta < self.theta:
                break

    def policy_improvement(self):
        for s in range(self.env.n_states):
            q = action_values(self.env, s, self.v, self.gamma)
            ties = np.isclose(q, q.max(), rtol=1e-12, atol=1e-12)
            self.pi[s] = ties / ties.sum()
        return self.pi

    def policy_iteration(self):
        while True:
            last = self.pi.copy()
            self.policy_evaluation()
            if np.allclose(self.policy_improvement(), last):
                return self.v, self.pi


class ValueIteration(PolicyIteration):
    def get_policy(self):
        return self.policy_improvement()

    def value_iteration(self):
        while True:
            new = np.array(
                [
                    action_values(self.env, s, self.v, self.gamma).max()
                    for s in range(self.env.n_states)
                ]
            )
            delta = np.max(abs(new - self.v))
            self.v = new
            if delta < self.theta:
                return self.v, self.get_policy()
