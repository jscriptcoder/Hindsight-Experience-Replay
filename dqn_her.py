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

gym_env = gym.make('LunarLander-v2')

BITS = 50
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1. # 0.95
EPOCHS = 30 # 200
CYCLES = 50
EPISODES = 16
OPTIMS = 50 # 40
MAX_STEPS = 300 # BITS
FUTURE_K = 4
STATE_SIZE = gym_env.observation_space.shape[0] * 2 # BITS * 2
ACTION_SIZE = gym_env.action_space.n # BITS
LR = 0.001
EPS_START = 0.2
EPS_END = 0.0
EPS_DECAY = 0.95
ENV_SOLVED = 0
TIMES_SOLVED = 100
EVAL_EVERY = 1

seed_all(0)



class BitFlipEnv:

    def __init__(self, bits):
        self.bits = bits
        self.state = np.zeros(bits)
        self.goal = np.zeros(bits)
        self.reset()

    def reset(self):
        self.state = np.random.randint(2, size=self.bits)
        self.goal = np.random.randint(2, size=self.bits)
        return np.copy(self.state), np.copy(self.goal)

    def step(self, action):
        self.state[action] = 1 - self.state[action]
        reward, done = self.compute_reward(self.state, self.goal)
        return np.copy(self.state), reward, done, {}

    def render(self):
        print('===')
        print('State:\t{}'.format(self.state.tolist()))
        print('Goal:\t{}'.format(self.goal.tolist()))
        print('===')
    
    def close(self):
        pass
    
    @staticmethod
    def compute_reward(state, goal):
        done = np.all(np.equal(state, goal))
        return 0. if done else -1., done

class LunarLanderEnv:
    def __init__(self, gym_env):
        self.env = gym_env
        self.goal = np.zeros(STATE_SIZE)

    def reset(self):
        self.goal = np.zeros(self.env.observation_space.shape[0])
        return self.env.reset(), np.copy(self.goal)

    def step(self, action):
        next_state, env_reward, env_done, info = self.env.step(action)
        reward, done = self.compute_reward(next_state, self.goal)
        return next_state, reward, (done or env_done), info

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    @staticmethod
    def compute_reward(state, goal, eps=0.1):
        done = np.sum(np.abs(goal - state)) < eps
        return 0. if done else -1., done



class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(256, action_size),
            
        )
        
        self.value = nn.Sequential(
            nn.Linear(256, 1)
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
    
    def compute_reward(self, state, goal, eps=0.1):
        done = np.sum(np.abs(goal - state)) < eps
        return 0. if done else -1., done
    
    def eval_episode(self, use_target=False, render=False):
        total_reward = 0
        
        for t in range(TIMES_SOLVED):
            state, goal = env.reset()
            
            for step in range(MAX_STEPS):
                
                if render and t == TIMES_SOLVED-1: env.render()
                    
                action = self.act(np.concatenate([state, goal]), use_target=use_target)
                # action = self.act(state, use_target=use_target)
                state, reward, done, _ = env.step(action)

                total_reward += reward
    
                if done: break
            
            if render: env.close()
                
        return total_reward / TIMES_SOLVED
    
    def train(self):
        print('Training on {}'.format(device))

        writer = SummaryWriter(comment='_LunarLander')
        eps = EPS_START

        for epoch in range(EPOCHS):
            
            success = 0
            
            for cycle in range(CYCLES):

                for episode in range(EPISODES):

                    trajectory = []
                    state, goal = env.reset()

                    for step in range(MAX_STEPS):
                        action = self.act(np.concatenate([state, goal]), eps)
                        # action = self.act(state, eps)

                        next_state, reward, done, _ = env.step(action)

                        trajectory.append(make_experience(state, action, reward, next_state, done))

                        state = next_state
                        
                        if done and reward == 0.: 
                            success += 1
                            break
                    # End Steps

                    steps_taken = len(trajectory)
                    for t in range(steps_taken):
                        state, action, reward, next_state, done = trajectory[t]
                        
                        self.add_experience(np.concatenate([state, goal]), 
                                            action, 
                                            reward, 
                                            np.concatenate([next_state, goal]), 
                                            done)

                        # self.add_experience(state, 
                        #                     action, 
                        #                     reward, 
                        #                     next_state, 
                        #                     done)

                        for _ in range(FUTURE_K):
                            future = np.random.randint(t, steps_taken)
                            achieved_goal = trajectory[future].next_state
                            reward, done = env.compute_reward(next_state, achieved_goal)
                            
                            self.add_experience(np.concatenate([state, achieved_goal]), 
                                                action, 
                                                reward, 
                                                np.concatenate([next_state, achieved_goal]), 
                                                done)
                        # End Goals
                    # End Steps
                # End Episode

                for _ in range(OPTIMS):
                    self.optimize()
                # End Optimization

                self.soft_update(TAU)
            # End Cycle
            
            success_rate = success / (EPISODES * CYCLES)
            print("epoch: {}, exploration: {:.0f}%, success rate: {:.2f}".format(epoch + 1, 100 * eps, success_rate))
            writer.add_scalar('Success Rate', success_rate, epoch)

            print('\nRunning evaluation...')

            mean_score = self.eval_episode(use_target=False, render=False)

            if mean_score >= ENV_SOLVED:
                print('Environment solved {} times consecutively!'.format(TIMES_SOLVED))
                print('Avg score: {:.3f}'.format(mean_score))
                break
            else:
                print('No success. Avg score: {:.3f}'.format(mean_score))
            
            eps = max(EPS_END, EPS_DECAY*eps)



# env = BitFlipEnv(BITS)
env = LunarLanderEnv(gym_env)

agent = DQNAgent()
agent.train()