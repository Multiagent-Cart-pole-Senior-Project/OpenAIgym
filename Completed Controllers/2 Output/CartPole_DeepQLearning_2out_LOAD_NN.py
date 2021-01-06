# Balancing a Single Cart-Pole using Deep Q-Learning
#   Author: Ryan Russell
#   Based on code from: 

import torch
import torch.nn as nn
import numpy as np
import random
import gym
from collections import deque

# PARAMETERS
GAMMA = 0.99
EPISODES = 1_000
BATCH_SIZE = 64
HIDDEN_DIM = 12
input_dim, output_dim = 4, 2

action_list = np.array([0, 1])

# PyTorch NN
class DQN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, HIDDEN_DIM: int):
        super(DQN, self).__init__()
        
        # Linear Layer 1
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, HIDDEN_DIM),
            torch.nn.BatchNorm1d(HIDDEN_DIM),
            torch.nn.PReLU()
        )
        
        # Linear Layer 2
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            torch.nn.BatchNorm1d(HIDDEN_DIM),
            torch.nn.PReLU()
        )
        
        # Ouput Layer
        self.final = torch.nn.Linear(HIDDEN_DIM, output_dim)
    
    # NN Function
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)

        return x

# Agent Object
class Agent(object):
    # Initialize NN for Agent
    def __init__(self, input_dim: int, output_dim: int, HIDDEN_DIM: int):
        self.dqn = DQN(input_dim, output_dim, HIDDEN_DIM)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())
        
    # Function to determine optimal action for Agent
    def get_action(self, X):
        NNinput = torch.autograd.Variable(torch.Tensor(X.reshape(-1, 4)))
        self.dqn.train(mode = False)
        qs = self.dqn(NNinput)
        _, argmax = torch.max(qs.data, 1)
        return int(argmax.numpy())

    # Function to train Agent's NN
    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor):
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()

        return loss

# Load Saved NN
agent = torch.load('net.pth')

# Set up Environment
env = gym.make("CartPole-v0")
rewards = deque(maxlen=100)
replay_memory = []
replay_time = 0

# Episode Loop
for episode in range(EPISODES):
    env.render()
    
    # Reset Environment
    obs = env.reset()
    done = False
    total_reward = 0
    
    # Simulation Loop
    while not done:
        # Determine Action
        a = agent.get_action(obs)
        
        # Take Action
        obs2, r, done, info = env.step(a)
        
        # Record Results
        total_reward += r
        if done:
            r = -1
        replay_memory.append([obs, a, r, obs2, done])
        replay_time += 1
        

        obs = obs2
    
    # Print Results of Episode
    r = total_reward   
    print("[Episode: {:5}] Reward: {:5}".format(episode + 1, r))
    
    rewards.append(r)


# Close the Environment
env.close()

# Save NN Model
torch.save(agent,'net.pth')
