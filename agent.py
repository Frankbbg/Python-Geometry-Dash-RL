import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

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
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        '''
        Add a new experience to the replay buffer.
        This works because:
        - state, action, reward, next_state, and done are all numpy arrays
        - the buffer is a list of tuples, where each tuple contains the 5 elements above
        '''
        if len(self.buffer) < self.capacity: # If the buffer is not full, add a new tuple
            self.buffer.append(None)
            
        # Add the new experience to the buffer by replacing the oldest experience
        self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Update the position to the next index
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
        Randomly sample a batch of experiences from the replay buffer.
        This works because:
        - random.sample returns a list of batch_size unique elements from the buffer
        - zip(*batch) transposes the batch list of tuples into a tuple of lists
        - map(np.stack, ...) converts each list of tuples into a list of numpy arrays
        '''
        batch = random.sample(self.buffer, batch_size) # Randomly sample a batch of experiences
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # Transpose the batch of experiences
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)