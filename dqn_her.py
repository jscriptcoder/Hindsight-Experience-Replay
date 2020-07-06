import gym
import time
import gym.spaces
import random
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as mpl
import rocket_lander_gym
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from common.replay_buffer import ReplayBuffer
from common.utils import make_experience, from_experience, seed_all
from common.device import device
 
warnings.filterwarnings('ignore')

env = gym.make('LunarLander-v2')

BUFFER_SIZE = int(1e5) # int(1e6)
BATCH_SIZE = 32 # 128
GAMMA = 0.98
TAU = 0.95
EPOCHS = 5 # 200
CYCLES = 50
EPISODES = 16
OPTIMS = 40
MAX_STEPS = 1000
FUTURE_K = 4
STATE_SIZE = env.observation_space.shape[0] # env.observation_space.shape[0] * 2
ACTION_SIZE = env.action_space.n
LR = 0.001
EPS_START = 0.2
EPS_END = 0.0
EPS_DECAY = 0.95
ENV_SOLVED = 200
TIMES_SOLVED = 100
EVAL_EVERY = 1

seed_all(0)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            
        )
        
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
            
    def forward(self, state):
        x = self.features(state)
        advantage = self.advantage(x)
        value = self.value(x)
        
        return value + advantage  - advantage.mean(dim=1, keepdim=True)

class DQNAgent:

    def __init__(self):
        self.qn_local = DuelingQNetwork(STATE_SIZE, 
                                        ACTION_SIZE).to(device)
        
        self.qn_target = DuelingQNetwork(STATE_SIZE, 
                                         ACTION_SIZE).to(device)
        
        self.soft_update(1.)
        
        self.optimizer = optim.Adam(self.qn_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    def act(self, state, eps=0., use_target=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        model = self.qn_target if use_target else self.qn_local
        
        model.eval()
        with torch.no_grad():
            action_values = model(state)
        
        model.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().numpy())
        else:
            return random.choice(np.arange(ACTION_SIZE))

    def optimize(self):
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            return self.learn(experiences)
            
    def learn(self, experiences):
        (states, 
         actions, 
         rewards, 
         next_states, 
         dones) = from_experience(experiences)
        
        best_action = self.qn_local(next_states).argmax(-1, keepdim=True)
        max_q = self.qn_target(next_states).detach().gather(-1, best_action)
        
        q_targets = rewards + (GAMMA * max_q * (1 - dones))
        q_expected = self.qn_local(states).gather(-1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.qn_target.parameters(), 
                                             self.qn_local.parameters()):
            target_param.data.copy_(tau * local_param + (1.0 - tau) * target_param)
    
    def add_experience(self, state, action, reward, next_state, done):
        experience = make_experience(state, 
                                     action, 
                                     reward, 
                                     next_state, 
                                     done)
        self.memory.add(experience)
    
    def make_goal(self):
        return np.array([0., 0., 0., 0., 0., 0., 1., 1.])
    
    def compute_reward(self, state, goal, eps=0.1):
        done = np.sum(np.abs(goal - state)) < eps
        return 0. if done else -1., done
    
    def eval_episode(self, use_target=False, render=False):
        total_reward = 0
        goal = self.make_goal()
        
        for t in range(TIMES_SOLVED):
            state = env.reset()
            
            for step in range(MAX_STEPS):
                
                if render and t == TIMES_SOLVED-1: env.render()
                    
                # action = self.act(np.concatenate([state, goal]), use_target=use_target)
                action = self.act(state, use_target=use_target)
                state, reward, done, _ = env.step(action)

                total_reward += reward
    
                if done: break
            
            if render: env.close()
                
        return total_reward / TIMES_SOLVED
    
    def train(self):
        print('Starting training...')

        writer = SummaryWriter(comment='DQN')
        eps = EPS_START

        episode_iter = 0
        optim_iter = 0
        cycle_iter = 0

        for epoch in range(1, EPOCHS+1):
            
            success = 0
            
            for cycle in range(1, CYCLES+1):
                
                total_reward = []

                for episode in range(1, EPISODES+1):

                    trajectory = []
                    state = env.reset()
                    goal = self.make_goal()

                    score = 0

                    for step in range(MAX_STEPS):
                        # action = self.act(np.concatenate([state, goal]), eps)
                        action = self.act(state, eps)
                        next_state, env_reward, env_done, _ = env.step(action)
                        # reward, done = self.compute_reward(next_state, goal)

                        # trajectory.append(make_experience(state, action, env_reward, next_state, done))
                        trajectory.append(make_experience(state, action, env_reward, next_state, env_done))

                        score += env_reward
                        state = next_state

                        # if done: success += 1
                        if env_done and env_reward == 100: success += 1
                        
                        if env_done: 
                            writer.add_scalar('Episode reward', env_reward, episode_iter)
                            episode_iter += 1
                            break

                    steps_taken = len(trajectory)
                    for t in range(steps_taken):
                        state, action, reward, next_state, done = trajectory[t]
                        
                        # self.add_experience(np.concatenate([state, goal]), 
                        #                     action, 
                        #                     reward, 
                        #                     np.concatenate([next_state, goal]), 
                        #                     done)

                        self.add_experience(state, 
                                            action, 
                                            reward, 
                                            next_state, 
                                            done)

                        # for _ in range(FUTURE_K):
                        #     future = np.random.randint(t, steps_taken)
                        #     achieved_goal = trajectory[future].next_state
                        #     reward, done = self.compute_reward(next_state, achieved_goal)
                            
                        #     self.add_experience(np.concatenate([state, achieved_goal]), 
                        #                         action, 
                        #                         reward, 
                        #                         np.concatenate([next_state, achieved_goal]), 
                        #                         done)                

                    total_reward.append(score)
 
                # End Episode

                print('\rEpoch {}, Explore: {:.2f}%, Cycle {}, Avg Reward: {:.3f}'.format(epoch, 100*eps, cycle, np.mean(total_reward)), end='')

                writer.add_scalar('Cycle success', success, cycle_iter)
                cycle_iter += 1

                for _ in range(OPTIMS):
                    loss = self.optimize()
                    if loss is not None:
                        writer.add_scalar('Optimization loss', loss, optim_iter)
                        optim_iter += 1
                # End Optimization

                self.soft_update(TAU)
            # End Cycle
            
            if epoch % EVAL_EVERY == 0:
                print('\nRunning evaluation...')

                mean_score = self.eval_episode(use_target=False, render=True)

                if mean_score >= ENV_SOLVED:
                    print('\tEnvironment solved {} times consecutively!'.format(TIMES_SOLVED))
                    print('\tAvg score: {:.3f}'.format(mean_score))
                    break
                else:
                    print('\tNo success. Avg score: {:.3f}'.format(mean_score))
            
            eps = max(EPS_END, EPS_DECAY*eps)

agent = DQNAgent()
agent.train()