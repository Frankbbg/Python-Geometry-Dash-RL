import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class GeometryDashCNN(nn.Module):
    def __init__(self):
        super(GeometryDashCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.fc1 = nn.Linear(64 * 7 * 4, 512) # original line
        self.fc1 = nn.Linear(3840, 512)  # Adjust 3840 to match the actual flattened size
        self.fc2 = nn.Linear(512, 2)  # Output: 2 actions (jump, not jump)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(9216, 512)  # Adjusted to the correct size
        self.fc2 = nn.Linear(512, 2)  # Output: 2 actions (jump, not jump)

    def forward(self, x):
        '''
        Forward pass for the actor network. A forward pass is the process of transforming the input data into an output
        using the neural network. In this case, the input is the state, and the output is the action.
        '''
        x = x.view(-1, 9216)  # Adjusting the view to match the new input size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(9216, 512)
        self.fc2 = nn.Linear(512, 1)  # Output: 1 value (state value)

    def forward(self, x):
        '''
        Forward pass for the critic network. A forward pass is the process of transforming the input data into an output 
        using the neural network. In this case, the input is the state, and the output is the state value.
        '''
        x = x.view(-1, 9216)  # Flatten the input tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PrioritizedReplayBuffer:
    def __init__(self, capacity, prob_alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_start + (1 - self.beta_start) * self.frame / self.beta_frames
        self.frame += 1

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -beta
        weights /= weights.max()

        return map(np.stack, zip(*samples)), indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)