import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import random
import gym
from collections import deque

# Parameters
T = 0.02
K1 = -23.5380
K2 = -5.1391
K3 = -0.7339
K4 = -1.2783

GAMMA = 0.99
EPISODES = 3_000
BATCH_SIZE = 64
MIN_EPS = 0.01
HIDDEN_DIM = 18
MAX_EPISODE = 200

action_list = np.array([-20, -10, 0, 10, 20]) # np.array([-10, -5, -3, -1, 0, 1, 3, 5, 10])

input_dim, output_dim = 4, 5

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

        # Linear Layer 3
        self.layer3 = torch.nn.Sequential(
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
        x = self.layer3(x)
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
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=0.001)
        
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

# Function to Organize and Modify Data for Training
def train_helper(agent, minibatch):
    # Take data from inputs
    states = np.vstack([x[0] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_states = np.vstack([x[3] for x in minibatch])
    done = np.array([x[4] for x in minibatch])
    
    # Create NNinput Tensors
    NNinput1 = torch.autograd.Variable(torch.Tensor(states.reshape(-1, 4)))
    NNinput2 = torch.autograd.Variable(torch.Tensor(next_states.reshape(-1, 4)))
    
    # Q-Value Updating
    Q_predict = agent.dqn(NNinput1)
    Q_target = Q_predict.clone().data.numpy()
    Q_target[np.arange(len(Q_target)), actions] = rewards + GAMMA * np.max(agent.dqn(NNinput2).data.numpy(), axis=1) * ~done
    Q_target = torch.autograd.Variable(torch.Tensor(Q_target))

    return agent.train(Q_predict, Q_target)


# Load Environment
env1 = gym.make('CartPole-v0')
env2 = gym.make('CartPole-v0')
rewards = deque(maxlen=100)
agent = Agent(input_dim, output_dim, HIDDEN_DIM)
replay_memory = []
replay_time = 0

# Episode Loop
for episode in range(EPISODES):

    theta_r = []
    theta_dot_r = []
    x_r = []
    x_dot_r = []
    
    x_fol = []
    theta_fol = []

    # Calculate epsilon for episode
    slope = (MIN_EPS - 1) / MAX_EPISODE
    eps = max(slope * episode + 1, MIN_EPS)
    
    # Reset Environment
    obs = env2.reset()
    observation = env1.reset()
    done = False
    total_reward = 0
    
    t = 0

    while not done:
        
        #
        # LINEAR CONTROLLER
        #
        
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
        
        # Calculate Control Input
        if t == 0:
            u = np.array([(-K1 * theta + -K2 * theta_dot + -K3 * x + -K4 * x_dot)])
        else:
            u = (-K1 * theta + -K2 * theta_dot + -K3 * x + -K4 * x_dot)
            
        t += 1
            
        # Apply Control Input
        observation, reward, done, info = env1.step(u)
                    
        # 
        # RL CONTROLLER
        #
        
        if episode > EPISODES * 2/3:
            env2.render()
        
        x_fol.append(obs[0])
        theta_fol.append(obs[2])
        
        # Determine Action
        if np.random.rand() > eps:
            a_num = agent.get_action(obs)
            a = action_list[a_num]
        else:
            a_num = np.random.randint(len(action_list))
            a = action_list[a_num]
            
        obs2, r, done, info = env2.step(a)
        
        # Calculate Reward
        if t > 0:
            r2 = - np.linalg.norm(np.array([2 * observation[0], observation[2]]) - np.array([2 * obs2[0], obs2[2]]))
        else: 
            r2 = 0
            
        r = r + r2
        
        # Record Results
        total_reward += r
        
        if done:
            r = -1000
            
        replay_memory.append([obs, a_num, r, obs2, done])
        replay_time += 1
        
        # Train after 64 timesteps with random sample of 64 from replay memory
        if len(replay_memory) > BATCH_SIZE: 
            minibatch = random.sample(replay_memory, BATCH_SIZE)
            train_helper(agent, minibatch)

        obs = obs2
            
    # Print Results of Episode
    r = total_reward   
    print("[Episode: {:5.2f}] Reward: {:5.2f} -greedy: {:5.2f} Time: {:5.2f}".format(episode + 1, r, eps, t))
    
    # Determine if the NN has learned to balance the system and end if so
    rewards.append(r)
    if len(rewards) == rewards.maxlen:

        if np.mean(rewards) >= 800:
            print("Game cleared in {} games with {}".format(episode + 1, np.mean(rewards)))
            break
 
# Close Environments
env1.close()
env2.close()

# Plot Outputs
plt.plot([t for t in range(len(x_r))],x_r)
plt.ylabel(f"x-position [m]")
plt.xlabel(f"Time Steps")
plt.show()

plt.plot([t for t in range(len(theta_r))],theta_r)
plt.ylabel(f"Pole Angle [rad]")
plt.xlabel(f"Time Steps")
plt.show()

plt.plot([t for t in range(len(x_fol))],x_fol)
plt.ylabel(f"x-position [m]")
plt.xlabel(f"Time Steps")
plt.show()

plt.plot([t for t in range(len(theta_fol))],theta_fol)
plt.ylabel(f"Pole Angle [rad]")
plt.xlabel(f"Time Steps")
plt.show()

plt.plot([t for t in range(len(x_fol))],abs(np.array(x_r) - np.array(x_fol)))
plt.ylabel(f"x-position Error [m]")
plt.xlabel(f"Time Steps")
plt.show()

plt.plot([t for t in range(len(theta_fol))],abs(np.array(theta_r) - np.array(theta_fol)))
plt.ylabel(f"Pole Angle Error [rad]")
plt.xlabel(f"Time Steps")
plt.show()