"""
DQN in PyTorch
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import gym
from collections import namedtuple
from collections import deque
from typing import List, Tuple

gamma = 0.99
n_episode = 1_000
batch_size = 64
replay_mem = 50_000
min_eps = 0.01
hidden_dim = 12
max_episode = 50

action_list = np.array([0, 1])

class DQN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)

        return x

class Agent(object):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())

    def get_action(self, X):
        """Returns an action
        Args:
            states (np.ndarray): 2-D tensor of shape (n, input_dim)
            eps (float): 𝜺-greedy for exploration
        Returns:
            int: action index
        """
        NNinput = torch.autograd.Variable(torch.Tensor(X.reshape(-1, 4)))
        self.dqn.train(mode = False) # DQNagent.train(mode=False)
        qs = self.dqn(NNinput) # DQNagent(NNinput)
        _, argmax = torch.max(qs.data, 1)
        return int(argmax.numpy()) # qs

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()

        return loss


def train_helper(agent: Agent, minibatch, gamma: float) -> float:
    """Prepare minibatch and train them
    Args:
        agent (Agent): Agent has `train(Q_pred, Q_true)` method
        minibatch (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of Q_target
    Returns:
        float: Loss value
    """
    states = np.vstack([x[0] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_states = np.vstack([x[3] for x in minibatch])
    done = np.array([x[4] for x in minibatch])
    
    NNinput1 = torch.autograd.Variable(torch.Tensor(states.reshape(-1, 4)))
    NNinput2 = torch.autograd.Variable(torch.Tensor(next_states.reshape(-1, 4)))

    Q_predict = agent.dqn(NNinput1) # agent.get_Q(states)
    Q_target = Q_predict.clone().data.numpy()
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(agent.dqn(NNinput2).data.numpy(), axis=1) * ~done
    Q_target = torch.autograd.Variable(torch.Tensor(Q_target))

    return agent.train(Q_predict, Q_target)



env = gym.make("CartPole-v0")
# env = gym.wrappers.Monitor(env, directory="monitors", force=True)
rewards = deque(maxlen=100)
input_dim, output_dim = 4, 2 #get_env_dim(env)
agent = Agent(input_dim, output_dim, hidden_dim)
replay_memory = [] # ReplayMemory(replay_mem)
replay_time = 0

for i in range(n_episode):
    # eps = epsilon_annealing(i, max_episode, min_eps)
    # r = play_episode(env, agent, replay_memory, eps, batch_size)
    
    # Calculate epsilon for episode
    slope = (min_eps - 1) / max_episode
    eps = max(slope * i + 1, min_eps)
    
    s = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Determine Action
        if np.random.rand() > eps:
            # current_qs_tensor = action(current_states)
            # # current_qs = current_qs_tensor.to("cpu").detach().numpy()
            # current_qs = current_qs_tensor.clone().data.numpy()
            # u = np.argmax(current_qs)
            a = agent.get_action(s)
        else:
            a = np.random.randint(len(action_list))
        s2, r, done, info = env.step(a)

        total_reward += r

        if done:
            r = -1
        replay_memory.append([s, a, r, s2, done]) # .push(s, a, r, s2, done)
        replay_time += 1

        if len(replay_memory) > batch_size: 

            # minibatch = replay_memory.pop(batch_size)
            minibatch = random.sample(replay_memory, batch_size)
            train_helper(agent, minibatch, gamma)

        s = s2
        
    r = total_reward   
    print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(i + 1, r, eps))

    rewards.append(r)

    if len(rewards) == rewards.maxlen:

        if np.mean(rewards) >= 200:
            print("Game cleared in {} games with {}".format(i + 1, np.mean(rewards)))
            break
env.close()
