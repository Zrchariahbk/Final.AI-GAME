import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import random
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)  
env = ResizeObservation(env, 84)    
env = GrayScaleObservation(env)     
env = FrameStack(env, 4)            

class MarioNet(nn.Module):
    """Neural network for Mario"""
    def __init__(self, input_shape, num_actions):
        super(MarioNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        ) 
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MarioAI:
    def __init__(self, state_dim, action_dim, save_path='mario_net.pth'):
        self.net = MarioNet(state_dim, action_dim).float().to(device)
        self.target_net = MarioNet(state_dim, action_dim).float().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.00025)
        self.memory = deque(maxlen=100000)
        self.save_path = save_path
        
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999999
        self.gamma = 0.9
        self.batch_size = 64
        
    def act(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.net(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        current_q = self.net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
       
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        
        self.target_net.load_state_dict(self.net.state_dict())
        
    def save(self):
        torch.save(self.net.state_dict(), self.save_path)


state_dim = env.observation_space.shape
action_dim = env.action_space.n
agent = MarioAI(state_dim, action_dim)


episodes = 100000
total_rewards = []
avg_rewards = []


for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total_reward += reward
        if done:
            break
            
    total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards[-100:])
    avg_rewards.append(avg_reward)
    print(f"Episode: {episode}, Total Reward: {total_reward}, Avg Reward (100): {avg_reward}, Epsilon: {agent.epsilon}")
    
    
    if episode % 50 == 0:
        agent.save()
        
    
    if episode % 10 == 0:
        plt.clf()
        plt.plot(total_rewards, label='Total Reward')
        plt.plot(avg_rewards, label='Average Reward (100)')
        plt.legend()
        plt.pause(0.001)

env.close()

