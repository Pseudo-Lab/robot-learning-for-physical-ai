import numpy as np
import torch
from torch.nn import functional as F
from ethz_starter.hw4_reinforcement_learning.exercises import ex2_dqn as course


class ReplayBuffer(course.ReplayBuffer):
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


class QNet(course.QNet):
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class DQN(course.DQN):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device,
    ):
        self.action_dim, self.gamma, self.epsilon = action_dim, gamma, epsilon
        self.device, self.target_update, self.count = device, target_update, 0
        self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        return self.predict_action(state)

    def update(self, transition_dict):
        def tensor(name, dtype=torch.float32):
            return torch.as_tensor(
                transition_dict[name], dtype=dtype, device=self.device
            )

        with torch.no_grad():
            future = self.target_q_net(tensor("next_states")).amax(dim=1)
            expected = tensor("rewards").reshape(-1) + self.gamma * future * (
                1 - tensor("dones").reshape(-1)
            )
        actions = tensor("actions", torch.long).reshape(-1, 1)
        estimates = self.q_net(tensor("states")).gather(1, actions).flatten()
        loss = F.mse_loss(estimates, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        return loss.item()
