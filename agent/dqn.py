import random
import numpy as np
import warnings
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from common.replay_buffer import ReplayBuffer
from common.utils import make_experience, from_experience, seed_all
from common.device import device
from common.scaler import StandarScaler, MinMaxScaler
from agent.network import DuelingQNetwork

class DQNAgent:

    def __init__(self, config):
        self.config = config

        lr = config.lr
        use_her = config.use_her
        state_size = config.state_size
        goal_size = config.goal_size
        action_size = config.action_size
        buffer_size = config.buffer_size
        batch_size = config.batch_size

        net_state_size = state_size + (goal_size if use_her else 0)
        self.qn_local = DuelingQNetwork(net_state_size, action_size).to(device)
        self.qn_target = DuelingQNetwork(net_state_size, action_size).to(device)
        
        self.soft_update(1.)
        
        self.optimizer = optim.Adam(self.qn_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        self.state_scaler = StandarScaler(state_size)
        self.goal_scaler = StandarScaler(goal_size)

    def process_input(self, state, goal=None):
        use_her = self.config.use_her

        state = self.state_scaler.scale(state)

        if use_her:
            goal = self.goal_scaler.scale(goal)
            return np.concatenate([state, goal])
        else:
            return state

    def act(self, state, goal, eps=0., use_target=False):
        action_size = self.config.action_size

        state_input = self.process_input(state, goal)
        state_input = torch.from_numpy(state_input).float().unsqueeze(0).to(device)
        
        model = self.qn_target if use_target else self.qn_local
        
        model.eval()
        with torch.no_grad():
            action_values = model(state_input)
        model.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().numpy())
        else:
            return random.choice(np.arange(action_size))

    def optimize(self):
        batch_size = self.config.batch_size

        if len(self.memory) > batch_size:
            experiences = self.memory.sample()
            return self.learn(experiences)
            
    def learn(self, experiences):
        (states, 
         actions, 
         rewards, 
         next_states, 
         dones) = from_experience(experiences)
        
        gamma = self.config.gamma
        use_her = self.config.use_her
        use_double = self.config.use_double
        use_huber_loss = self.config.use_huber_loss
        
        if use_double:
            best_action = self.qn_local(next_states).argmax(-1, keepdim=True)
            max_q = self.qn_target(next_states).detach().gather(-1, best_action)
        else:
            max_q = self.qn_target(next_states).detach().max(-1, keepdim=True)[0]
        
        q_targets = rewards + (gamma * max_q * (1 - dones))

        if use_her:
            clip_return = 1 / (1 - gamma)
            q_targets = torch.clamp(q_targets, -clip_return, 0)

        q_expected = self.qn_local(states).gather(-1, actions)

        if use_huber_loss:
            loss = F.smooth_l1_loss(q_expected, q_targets)
        else:
            loss = F.mse_loss(q_expected, q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.qn_target.parameters(), 
                                             self.qn_local.parameters()):
            target_param.data.copy_(tau * local_param + (1.0 - tau) * target_param)
    
    def add_experience(self, state, action, reward, next_state, done, goal=None):
        use_her = self.config.use_her

        self.state_scaler.update(state)
        if use_her: self.goal_scaler.update(goal)

        state_ = self.process_input(state, goal)
        next_state_ = self.process_input(next_state, goal)

        experience = make_experience(state_, 
                                     action, 
                                     reward, 
                                     next_state_, 
                                     done, 
                                     {})
        self.memory.add(experience)

    def eval_episode(self, env, use_target=False, render=False):
        times_solved = self.config.times_solved
        max_steps = self.config.max_steps

        total_reward = 0
        
        for t in range(times_solved):
            state, goal = env.reset()
            
            for step in range(max_steps):
                
                if render and t == times_solved-1: env.render()
                
                action = self.act(state, goal, use_target=use_target)
                state, reward, done, info = env.step(action)

                total_reward += info['env_reward']
    
                if done: break
            
            if render: env.close()
                
        return total_reward / times_solved
    
    def train(self, env):
        print('Training on {}'.format(device))

        tau = self.config.tau
        eps = self.config.eps_start
        eps_end = self.config.eps_end
        eps_decay = self.config.eps_decay
        epochs = self.config.epochs
        cycles = self.config.cycles
        episodes = self.config.episodes
        max_steps = self.config.max_steps
        use_her = self.config.use_her
        future_k = self.config.future_k
        optims = self.config.optims
        env_solved = self.config.env_solved
        times_solved = self.config.times_solved

        writer = SummaryWriter(comment='_LunarLander')

        for epoch in range(epochs):

            rewards = []
            
            for cycle in range(cycles):

                for episode in range(episodes):

                    trajectory = []
                    state, goal = env.reset()

                    for step in range(max_steps):
                        action = self.act(state, goal, eps)
                        next_state, reward, done, info = env.step(action)

                        self.add_experience(state, 
                                            action, 
                                            reward, 
                                            next_state, 
                                            done, 
                                            goal)
                        
                        trajectory.append(make_experience(state, 
                                                          action, 
                                                          reward, 
                                                          next_state, 
                                                          done, 
                                                          info))

                        rewards.append(reward)
                        state = next_state
                        
                        if done: break
                    # End Steps
                    
                    if use_her:
                        steps_taken = len(trajectory)
                        
                        if future_k == 1:
                            # final stragegy
                            goals_idx = [steps_taken-1]
                        else: 
                            # future strategy
                            goals_idx = np.random.choice(steps_taken, future_k, replace=False)

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

                for _ in range(optims):
                    self.optimize()
                # End Optimization

                self.soft_update(tau)
            # End Cycle
            
            avg_reward = np.mean(rewards)

            print('epoch: {}, exploration: {:.2f}%, avg score: {:.3f}'.format(epoch + 1, 100 * eps, avg_reward))
            writer.add_scalar('Avg Score', avg_reward, epoch)

            if abs(avg_reward) <= 0.0002:
                print('\nRunning evaluation...')

                mean_score = self.eval_episode(env, use_target=False, render=True)
                writer.add_scalar('Evaluation Score', mean_score, epoch)

                if mean_score >= env_solved:
                    print('Environment solved {} times consecutively!'.format(times_solved))
                    print('Avg score: {:.3f}'.format(mean_score))
                    break
                else:
                    print('No success. Avg score: {:.3f}'.format(mean_score))
            
            eps = max(eps_end, eps_decay*eps)
        # End Epoch
