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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

# Parameters
T = 0.02
SIM_TIME = 100 # Steps

K1 = -23.5380
K2 = -5.1391
K3 = -0.7339
K4 = -1.2783

REPLAY_MEMORY = 1_000
TARGET_UPDATE = 5
MINI_BATCH_SIZE = 64
LEARNING_RATE = 0.1
DISCOUNT = 0.90
EPOCHS = 100_000

action_list = np.arange(-1, 1, 0.05)


# Select device to use
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Create NN for Agent 1
class Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, len(action_list))
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)
        
def fwd_train(X_1, y_1, MINI_BATCH_SIZE):
    # Train Agent 1
    for i in range(MINI_BATCH_SIZE):
        X_1t = torch.Tensor(np.array(X_1[i])).to(device)
        y_1t = torch.Tensor(np.array(y_1[i])).to(device)
        print(X_1t.shape)
        net_1.zero_grad()
        outputs_1 = net_1(X_1t)
        loss_1 = loss_function(outputs_1, y_1t)
        loss_1.backward()
        optimizer_1.step()
    
    return loss_1

# Create NNs
net_1 = Model_1().to(device)
net_1_pred = copy.deepcopy(net_1)
target_counter = 0

optimizer_1 = optim.Adam(net_1.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

current_size = 0
replay_memory = []
states = []
new_states = []
action = []
rewards = []

reward_sum = []

# Load Environment
env = gym.make('CartPole-v0')
observation = env.reset()

# Simulation
for EPOCH in tqdm(range(EPOCHS)):

    # Data to Record
    theta_r = []
    theta_dot_r = []
    x_r = []
    x_dot_r = []
    
    observation = env.reset()
    
    reward_sum = 0
    for t in range(SIM_TIME):
    
        # Get observations
        theta = observation[2]
        theta_dot = observation[3]
        x = observation[0]
        x_dot = observation[1]   

        # Save observations for recording
        theta_r.append(theta)
        theta_dot_r.append(theta_dot)
        x_r.append(x)
        x_dot_r.append(x_dot)
        
        if current_size >= REPLAY_MEMORY:
            mini_batch = random.sample(replay_memory, MINI_BATCH_SIZE)
            
            action_batch = []
            for i in range(MINI_BATCH_SIZE):
                action_batch.append(action[mini_batch[i]])            
            reward_batch = []
            for i in range(MINI_BATCH_SIZE):
                reward_batch.append(rewards[mini_batch[i]])
            current_states = []
            for i in range(MINI_BATCH_SIZE):
                current_states.append(states[mini_batch[i]])
            current_qs = []
            for i in range(MINI_BATCH_SIZE):
                temp = np.array(current_states[i])
                temp1 = net_1(torch.Tensor(temp).to(device)).to("cpu").detach().numpy()
                current_qs.append(temp1)
                
            new_states_batch = []
            for i in range(MINI_BATCH_SIZE):
                new_states_batch.append(new_states[mini_batch[i]])
            future_qs = []
            for i in range(MINI_BATCH_SIZE):
                temp = np.array(new_states[i])
                temp1 = net_1_pred(torch.Tensor(temp).to(device)).to("cpu").detach().numpy()
                future_qs.append(temp1)
            
            X_1 = []
            y_1 = []
            
            for i in range(MINI_BATCH_SIZE):
                max_future_q = np.max(future_qs[i])
                new_q = reward_batch[i] + DISCOUNT * max_future_q
                current_q = current_qs[i]
                current_q[action_batch[i]] = new_q
                
                X_1.append(current_states[i])
                y_1.append(current_q)
                
            # Train the NN
            fwd_train(X_1, y_1, MINI_BATCH_SIZE)
            
            # See if need to update Target Model_1
            if target_counter > TARGET_UPDATE:
                net_1_pred = copy.deepcopy(net_1)
                target_counter = 0
            else:
                target_counter = target_counter + 1
            
            # Reset Variables after Training
            current_size = 0
            states = []
            new_states = []
            rewards = []
            action = []
            replay_memory = []
            
        else:
            replay_memory.append(current_size)
            states.append([theta, theta_dot, x, x_dot])
            current_size = current_size + 1
            
            # Calculate Control Input
            current_state = [theta, theta_dot, x, x_dot]
            qs = net_1(torch.Tensor(np.array(current_state)).to(device))
            
            qs = qs.to("cpu")
            
            current_action = np.argmax(qs.detach().numpy())
            
            if t == 0:
                u = np.array(action_list[current_action])
            else:
                u = action_list[current_action]
                
            # Apply Control Input
            observation, reward, done, info = env.step(u)
            
            # Get new observations
            theta = observation[2]
            theta_dot = observation[3]
            x = observation[0]
            x_dot = observation[1]  
            
            new_states.append([theta, theta_dot, x, x_dot])
            rewards.append(reward)
            action.append(current_action)
            
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
        
# Close Environment
env.close()

# Plot Outputs
plt.plot([t for t in range(len(x_r))],x_r)
plt.ylabel(f"x-position [m]")
plt.xlabel(f"Time Steps")
plt.show()

plt.plot([t for t in range(len(theta_r))],theta_r)
plt.ylabel(f"Pole Angle [rad]")
plt.xlabel(f"Time Steps")
plt.show()