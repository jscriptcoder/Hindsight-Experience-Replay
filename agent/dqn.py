import random
import numpy as np
import warnings
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from common.replay_buffer import ReplayBuffer
from common.utils import make_experience, from_experience, seed_all, sample_goals_idx
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
        
        self.scaler = StandarScaler(net_state_size)
        self.losses = deque(maxlen=100)

    def process_input(self, state, goal=None):
        use_her = self.config.use_her

        if use_her and goal is not None:
            state =  np.concatenate([state, goal])
            # state = self.scaler.scale(state)
        
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
        
        tau = self.config.tau
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
        
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(tau)
    
    def soft_update(self, tau):
        for target_param, local_param in zip(self.qn_target.parameters(), 
                                             self.qn_local.parameters()):
            target_param.data.copy_(tau * local_param + (1.0 - tau) * target_param)
    
    def add_experience(self, state, action, reward, next_state, done, goal=None):
        use_her = self.config.use_her

        if use_her and goal is not None:
            self.scaler.update(np.concatenate([state, goal]))
            self.scaler.update(np.concatenate([next_state, goal]))

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
        max_steps = self.config.max_steps

        total_reward = 0
        state, goal = env.reset()
        done = False
        
        for _ in range(max_steps):
            
            if render: env.render()
            
            action = self.act(state, goal, use_target=use_target)
            state, reward, done, info = env.step(action)

            total_reward += reward
        
        if render: env.close()
                
        return total_reward
    
    def train(self, env):
        print('Training on {}'.format(device))

        tau = self.config.tau
        eps = self.config.eps_start
        eps_end = self.config.eps_end
        eps_decay = self.config.eps_decay
        episodes = self.config.episodes
        max_steps = self.config.max_steps
        use_her = self.config.use_her
        future_k = self.config.future_k
        eval_every = self.config.eval_every

        writer = SummaryWriter(comment='_LunarLander')

        best_eval_score = -np.inf

        for episode in range(episodes):

            total_reward = 0
            trajectory = []
            state, goal = env.reset()

            for _ in range(max_steps):
                action = self.act(state, goal, eps)
                next_state, reward, done, info = env.step(action)

                self.add_experience(state, 
                                    action, 
                                    reward, 
                                    next_state, 
                                    done, 
                                    goal)
                
                self.optimize()

                trajectory.append(make_experience(state, 
                                                  action, 
                                                  reward, 
                                                  next_state, 
                                                  done, 
                                                  info))

                total_reward += reward
                state = next_state
                
                if done: 
                    break
            
            if use_her:
                steps_taken = len(trajectory)
                
                goals_idx = sample_goals_idx(steps_taken, future_k)

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
                        
                        self.optimize()
                # End Goals
            
            avg_loss = np.mean(self.losses)

            print('episode: {}, exploration: {:.2f}%, total reward: {:.3f}, avg Loss: {:.3f}'.format(episode + 1, 100 * eps, total_reward, avg_loss))
            writer.add_scalar('Episode Reward', total_reward, episode)
            writer.add_scalar('Avg Loss', avg_loss, episode)

            # if success:
            if (episode+1) % eval_every == 0:
                print('\nRunning evaluation...')

                score = self.eval_episode(env, use_target=False, render=False)

                print('eval score: {:.3f}\n'.format(score))
                writer.add_scalar('Evaluation Score', score, episode)

                if score > best_eval_score:
                    best_eval_score = score
                    torch.save(self.qn_local.state_dict(), 'best_weights.pth')
                
            eps = max(eps_end, eps_decay*eps)
