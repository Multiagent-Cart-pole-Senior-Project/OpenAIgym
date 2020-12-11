import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
T = 0.02
K1 = -23.5380
K2 = -5.1391
K3 = -0.7339
K4 = -1.2783
amplitude = 0.05
w = 0.5

theta_r = []
theta_dot_r = []
x_r = []
x_dot_r = []

# Load Environment
env = gym.make('CartPole-v0')
observation = env.reset()
for t in tqdm(range(1000)):
    env.render()
    
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
        u = np.array([(-K1 * theta + -K2 * theta_dot + -K3 * x + -K4 * x_dot) + amplitude*np.sin(w*T*t)])
    else:
        u = (-K1 * theta + -K2 * theta_dot + -K3 * x + -K4 * x_dot) + amplitude*np.sin(w*T*t)
        
    # Apply Control Input
    observation, reward, done, info = env.step(u)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
 
# Close Environment
env.close()

# Plot Outputs
plt.plot([t for t in range(1000)],x_r)
plt.ylabel(f"x-position [m]")
plt.xlabel(f"Time Steps")
plt.show()

plt.plot([t for t in range(1000)],theta_r)
plt.ylabel(f"Pole Angle [rad]")
plt.xlabel(f"Time Steps")
plt.show()
