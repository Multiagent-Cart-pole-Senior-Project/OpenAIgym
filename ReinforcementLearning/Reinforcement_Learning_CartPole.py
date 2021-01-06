# Deep Q-Learning to Solve Open AI CartPolev0
# Author: Ryan Russell

# From other - See for deletion later
import argparse
from collections import namedtuple
from collections import deque
from typing import List, Tuple

# My Imports
import gym
import math
import random
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

# HyperParameters
GAMMA = 0.99
EPISODES = 1_000
BATCH_SIZE = 64
REPLAY_MEM = 50_000
EPSILON_MIN = 0.01
MAX_EP = 50
LEARNING_RATE = 0.001

# Possible Actions
action_list = np.array([0, 1])

# # Determine if on GPU or CPU
# if torch.cuda.is_available():
    # device = torch.device("cuda:0")
    # print("Running on the GPU")
# else:
    # device = torch.device("cpu")
    # print("Running on the CPU")
    
# device = torch.device("cpu")

# Create NN
class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # NN Layers
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(4, 12),
            torch.nn.BatchNorm1d(12),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(12, 12),
            torch.nn.BatchNorm1d(12),
            torch.nn.PReLU()
        )
        
        self.outLayer = torch.nn.Linear(12, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.outLayer(x)
        
        return x
        

class Agent(object):

    def __init__(self):

        self.dqn = DQN()
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())
        
    # Determine Action to take (Input: Numpy Array of States, Output: Tensor of Q-values)        
    def action(self, X):
        NNinput = torch.autograd.Variable(torch.Tensor(X.reshape(-1, 4)))
        self.dqn.train(mode = False) # DQNagent.train(mode=False)
        qs = self.dqn(NNinput) # DQNagent(NNinput)
        _, argmax = torch.max(qs.data, 1)
        return int(argmax.numpy()) # qs

    # Train NN (Input: Q_pred and Q_new Tensors, Output: Loss)    
    def train(self, Q_pred, Q_new):
        # self.dqn.train(mode = True) # DQNagent.train(mode=True)
        # optimizer.zero_grad()
        # loss = loss_func(Q_pred, Q_new)
        # loss.backward()
        # optimizer.step()
        
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_new)
        loss.backward()
        self.optim.step()
        
        return loss

# Train Helper (Input: mini_batch list, Output: Loss)
def train_help(mini_batch):
    # Q_pred_list = []
    # Q_new_list = []
    
    # Calcualte Q_Pred and Q_new
    # for i in range(BATCH_SIZE):
        # replay_instance = mini_batch[i]
        # rew = replay_instance[3]
        # act = replay_instance[1]
        # c_qs = [replay_instance[4][0][0], replay_instance[4][0][1]]
        # n_qs = [replay_instance[5][0][0], replay_instance[5][0][1]]
        # dn = replay_instance[6]
        
        # NNinput = torch.autograd.Variable(torch.Tensor(np.array(replay_instance[0]).reshape(-1, 4)))
        # Q_p = DQNagent(NNinput)
        # Q_n = Q_p.clone().data.numpy()
        # print(Q_n)
        # Q_n[act] = rew + GAMMA * np.max(n_qs) * ~dn
        
        # Q_pred_list.append(Q_p)
        # Q_new_list.append(Q_n)
        
    # Q_pred = torch.Tensor(np.array(Q_pred_list)).to(device)
    # Q_new = torch.Tensor(np.array(Q_new_list)).to(device)
    
    # replay_states = []
    # replay_new_states = []
    # rew = []
    # dn = []
    # act = []
    # for i in range(BATCH_SIZE):
        # replay_states.append(mini_batch[i][0])
        # replay_new_states.append(mini_batch[i][2])
        # rew.append(mini_batch[i][3])
        # dn.append(mini_batch[i][4])
        # act.append(mini_batch[i][1])
        
        
    
    # rew = np.array(rew)
    # dn = np.array(dn)
    # act = np.array(act)
    
    replay_states = np.vstack([x[0] for x in mini_batch])
    act = np.array([x[1] for x in mini_batch])
    rew = np.array([x[3] for x in mini_batch])
    replay_new_states = np.vstack([x[2] for x in mini_batch])
    dn = np.array([x[4] for x in mini_batch])
        
    NNinput1 = torch.autograd.Variable(torch.Tensor(np.array(replay_states).reshape(-1, 4)))
    NNinput2 = torch.autograd.Variable(torch.Tensor(np.array(replay_new_states).reshape(-1, 4)))
    
    Q_p = agent.dqn(NNinput1) # DQNagent(NNinput1)
    Q_n = Q_p.clone().data.numpy()
    Q_n[np.arange(len(Q_n)), act] = rew + GAMMA * np.max(agent.dqn(NNinput2).data.numpy(), axis = 1) * ~dn # np.max(DQNagent(NNinput2).data.numpy(), axis = 1) * ~dn
    Q_new = torch.Tensor(Q_n)# .to(device)
    
    # Q_pred.requires_grad_(True)
    # Q_new.requires_grad_(True)
    
    Q_pred = Q_p
    Q_new = torch.autograd.Variable(Q_new)
    
    return agent.train(Q_pred, Q_new)
    
# Initialize Environment
env = gym.make('CartPole-v0')

# Initialize Agent
# DQNagent = DQN().to(device)
# loss_func = nn.MSELoss()
# optimizer = optim.Adam(DQNagent.parameters())

agent = Agent()

# Initialize Variables
episode_rewards = deque(maxlen=100)
replay_memory = []
replay_time = 0
loss_record = []


# Episode Loop
for episode in range(EPISODES):
    
    # Calculate epsilon for episode
    s = (EPSILON_MIN - 1) / MAX_EP
    epsilon = max(s * episode + 1, EPSILON_MIN)
    
    # Reset before episode
    observation = env.reset()
    done = False
    total_episode_reward = 0
    
    # Episoded Simulation Loop
    while not done:
        # Format State to Numpy Array (x, xdot, theta, theta_dot)
        current_states = np.array([observation[0], observation[1], observation[2], observation[3]])
    
        # Determine Action
        if np.random.rand() > epsilon:
            # current_qs_tensor = action(current_states)
            # # current_qs = current_qs_tensor.to("cpu").detach().numpy()
            # current_qs = current_qs_tensor.clone().data.numpy()
            # u = np.argmax(current_qs)
            u = agent.action(current_states)
        else:
            u = np.random.randint(len(action_list))

        # Take Action
        observation, reward, done, info = env.step(u)
        
        new_states = np.array([observation[0], observation[1], observation[2], observation[3]])
        
        # new_qs_tensor = action(new_states)
        # # new_qs = new_qs_tensor.to("cpu").detach().numpy()        
        # new_qs = new_qs_tensor.clone().data.numpy()
       
        total_episode_reward += reward
        
        
        if done:
            reward = -1
        
        # Record Results
        replay_memory.append([current_states, u, new_states, reward, done])
        replay_time += 1
        
        # print(replay_time)
        
        # Train NN
        if len(replay_memory) > BATCH_SIZE:
            current_replay = replay_memory[-BATCH_SIZE:]
            
            # Sample into minibatch
            mini_batch = random.sample(replay_memory, BATCH_SIZE) # current_replay, BATCH_SIZE)
            # mini_batch = current_replay

            loss5 = train_help(mini_batch)
            loss_record.append(loss5)
            replay_time = 0
            
    print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(episode + 1, total_episode_reward, epsilon))

# Close Environment
env.close()

# Plot Results



        