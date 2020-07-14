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
from common.lunar_lander_goal import LunarLanderGoal
from common.normalizer import Normalizer
 
warnings.filterwarnings('ignore')

gym_env = gym.make('LunarLander-v2')

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.98
TAU = 0.95
EPOCHS = 200
CYCLES = 50
EPISODES = 16
OPTIMS = 40
MAX_STEPS = 500
FUTURE_K = 4
STATE_SIZE = gym_env.observation_space.shape[0]
ACTION_SIZE = gym_env.action_space.n
GOAL_SIZE = STATE_SIZE-2
LR = 0.001
EPS_START = 0.2
EPS_END = 0.01
EPS_DECAY = 0.9995
ENV_SOLVED = 200
TIMES_SOLVED = 100

seed_all(0)

env = LunarLanderGoal(gym_env)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            
        )
        
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
            
    def forward(self, state):
        x = self.features(state)
        advantage = self.advantage(x)
        value = self.value(x)
        
        return value + advantage  - advantage.mean(dim=1, keepdim=True)

class DQNAgent:

    def __init__(self):
        self.qn_local = DuelingQNetwork(STATE_SIZE+GOAL_SIZE, 
                                        ACTION_SIZE).to(device)
        
        self.qn_target = DuelingQNetwork(STATE_SIZE+GOAL_SIZE, 
                                         ACTION_SIZE).to(device)
        
        self.soft_update(1.)
        
        self.optimizer = optim.Adam(self.qn_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.state_norm = Normalizer(STATE_SIZE)
        self.goal_norm = Normalizer(GOAL_SIZE)

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
        # clip_return = 1 / (1 - GAMMA)
        # q_targets = torch.clamp(q_targets, -clip_return, 0)

        q_expected = self.qn_local(states).gather(-1, actions)

        loss = F.smooth_l1_loss(q_expected, q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.qn_target.parameters(), 
                                             self.qn_local.parameters()):
            target_param.data.copy_(tau * local_param + (1.0 - tau) * target_param)
    
    def process_input(self, state, goal):
        state = self.state_norm.normalize(state)
        goal = self.goal_norm.normalize(goal)
        return np.concatenate([state, goal])
        # return state
    
    def add_experience(self, state, action, reward, next_state, done, goal):
        self.state_norm.update(state)
        self.goal_norm.update(goal)

        state_ = self.process_input(state, goal)
        next_state_ = self.process_input(next_state, goal)

        experience = make_experience(state_, 
                                     action, 
                                     reward, 
                                     next_state_, 
                                     done, 
                                     {})
        self.memory.add(experience)

    def eval_episode(self, use_target=False, render=False):
        total_reward = 0
        
        for t in range(TIMES_SOLVED):
            state, goal = env.reset()
            
            for step in range(MAX_STEPS):
                
                if render and t == TIMES_SOLVED-1: env.render()
                
                state_input = self.process_input(state, goal)
                action = self.act(state_input, use_target=use_target)
                state, reward, done, info = env.step(action)

                total_reward += info['env_reward']
    
                if done: break
            
            if render: env.close()
                
        return total_reward / TIMES_SOLVED
    
    def train(self):
        print('Training on {}'.format(device))

        writer = SummaryWriter(comment='_LunarLander')
        eps = EPS_START

        for epoch in range(EPOCHS):
            
            successes = []
            total_rewards = []
            
            for cycle in range(CYCLES):

                for episode in range(EPISODES):

                    trajectory = []
                    state, goal = env.reset()
                    
                    score = 0

                    for step in range(MAX_STEPS):
                        state_input = self.process_input(state, goal)
                        action = self.act(state_input, eps)

                        next_state, reward, done, info = env.step(action)

                        self.add_experience(state, 
                                            action, 
                                            reward, 
                                            next_state, 
                                            done, 
                                            goal)
                        
                        trajectory.append(make_experience(state, action, reward, next_state, done, info))

                        state = next_state

                        score += info['env_reward']

                        if done: break
                    # End Steps

                    total_rewards.append(score)
                    successes.append(info['success'])
                    
                    steps_taken = len(trajectory)
                    goals_idx = np.random.choice(steps_taken, FUTURE_K, replace=False)

                    for goal_i in goals_idx:
                        new_goal = trajectory[goal_i].info['achieved_goal']

                        for t in range(goal_i+1):
                            state, action, reward, next_state, done, info = trajectory[t]

                            achieved_goal = trajectory[t].info['achieved_goal']
                            reward, done = env.compute_reward(achieved_goal, new_goal)

                            self.add_experience(state, 
                                                action, 
                                                reward, 
                                                next_state, 
                                                done, 
                                                new_goal)

                            if done: break
                        # End HER
                    # End Goals
                # End Episode

                losses = []
                for _ in range(OPTIMS):
                    loss = self.optimize()
                    if loss is not None: losses.append(loss)
                # End Optimization

                self.soft_update(TAU)
            # End Cycle
            
            success_rate = np.mean(successes)
            avg_score = np.mean(total_rewards)
            avg_loss = np.mean(losses)

            print("epoch: {}, exploration: {:.0f}%, success rate: {:.3f}, avg score: {:.3f}, avg loss: {:.3f}".format(epoch + 1, 100 * eps, success_rate, avg_score, avg_loss))

            writer.add_scalar('Success Rate', success_rate, epoch)
            writer.add_scalar('Avg Score', avg_score, epoch)
            writer.add_scalar('Avg Loss', avg_loss, epoch)

            print('\nRunning evaluation...')

            mean_score = self.eval_episode(use_target=False, render=False)

            writer.add_scalar('Evaluation Score', mean_score, epoch)

            if mean_score >= ENV_SOLVED:
                print('Environment solved {} times consecutively!'.format(TIMES_SOLVED))
                print('Avg score: {:.3f}'.format(mean_score))
                break
            else:
                print('No success. Avg score: {:.3f}'.format(mean_score))
            
            eps = max(EPS_END, EPS_DECAY*eps)
        # End Epoch

agent = DQNAgent()

agent.train()