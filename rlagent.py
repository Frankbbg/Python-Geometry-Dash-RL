import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class GeometryDashCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(GeometryDashCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _feature_size(self, input_shape):
        return torch.autograd.Variable(torch.zeros(1, *input_shape)).view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.primary_network = GeometryDashCNN(state_dim, action_dim).to(device)
        self.target_network = GeometryDashCNN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.primary_network.state_dict())
        self.optimizer = torch.optim.Adam(self.primary_network.parameters(), lr=1e-4)

    def update_target_network(self):
        self.target_network.load_state_dict(self.primary_network.state_dict())

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.primary_network(state)
        return q_values.max(1)[1].item()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = zip(*self.replay_buffer.sample(batch_size))
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q_values = self.primary_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        expected_q_values = rewards + 0.99 * next_q_values * (1 - dones)

        loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()