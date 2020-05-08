import time
import random
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .actor import Actor
from .critic import Critic
from .noise import OUNoise, GaussianNoise
from .utils import soft_update, make_experience, from_experience, get_time_elapsed
from .device import device


class DDPGAgent():
    """Deep Deterministic Policy Gradient
    https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
    """

    name = 'DDPG'

    def __init__(self, config):
        self.config = config

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(config.state_size,
                                 config.action_size,
                                 config.hidden_actor,
                                 config.activ_actor)

        self.actor_target = Actor(config.state_size,
                                  config.action_size,
                                  config.hidden_actor,
                                  config.activ_actor)

        soft_update(self.actor_local, self.actor_target, 1.0)

        self.actor_optim = config.optim_actor(self.actor_local.parameters(),
                                              lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(config.state_size,
                                   config.action_size,
                                   config.hidden_critic,
                                   config.activ_critic)

        self.critic_target = Critic(config.state_size,
                                    config.action_size,
                                    config.hidden_critic,
                                    config.activ_critic)

        soft_update(self.critic_local, self.critic_target, 1.0)

        self.critic_optim = config.optim_critic(self.critic_local.parameters(),
                                                lr=config.lr_critic)

        # Noise process
        if config.use_ou_noise:
            self.noise = OUNoise(config.action_size,
                                 config.ou_mu,
                                 config.ou_theta,
                                 config.ou_sigma)
        else:
            self.noise = GaussianNoise(config.action_size, config.expl_noise)

        self.noise_weight = config.noise_weight
        self.t_step = 0
        self.p_update = 0

    def act(self, state, add_noise=True):
        decay_noise = self.config.decay_noise
        use_linear_decay = self.config.use_linear_decay
        noise_linear_decay = self.config.noise_linear_decay
        noise_decay = self.config.noise_decay

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample() * self.noise_weight

            if decay_noise:
                if use_linear_decay:
                    self.noise_weight = max(0.1,
                                            self.noise_weight - noise_linear_decay)
                else:
                    self.noise_weight = max(0.1,
                                            self.noise_weight * noise_decay)

        return np.clip(action, -1., 1.)

    def reset(self):
        self.noise.reset()
        return self.config.env.reset()

    def step(self, state, action, reward, next_state, done):
        batch_size = self.config.batch_size
        update_every = self.config.update_every
        num_updates = self.config.num_updates

        experience = make_experience(state,
                                     action,
                                     reward,
                                     next_state,
                                     done)
        self.memory.add(experience)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % update_every

        if self.t_step == 0:

            # Learn, if enough samples are available in memory
            if len(self.memory) > batch_size:

                # Multiple updates in one learning step
                for _ in range(num_updates):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def learn(self, experiences):

        policy_freq_update = self.config.policy_freq_update
        batch_size = self.config.batch_size

        (states,
         actions,
         rewards,
         next_states,
         dones) = from_experience(experiences)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target(next_states)

        self.update_critic(states.view(batch_size, -1),
                           actions.view(batch_size, -1),
                           next_states.view(batch_size, -1),
                           next_actions,
                           rewards.view(-1, 1),
                           dones.view(-1, 1))

        self.p_update = (self.p_update + 1) % policy_freq_update

        if self.p_update == 0:
            # ---------------------------- update actor ---------------------------- #
            pred_actions = self.actor_local(states)

            self.update_actor(states.view(batch_size, -1),
                              pred_actions)

    def update_critic(self,
                      states,
                      actions,
                      next_states,
                      next_actions,
                      rewards,
                      dones):

        grad_clip_critic = self.config.grad_clip_critic
        use_huber_loss = self.config.use_huber_loss
        gamma = self.config.gamma
        tau = self.config.tau

        # next_states => tensor(batch_size, 48)
        # next_actions => tensorbatch_size, 4)
        # Q_targets_next => tensor(batch_size, 1)
        Q_targets_next = \
            self.critic_target(next_states, next_actions).detach()

        # Compute Q targets for current states
        # tensor(batch_size, 1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # states => tensor(batch_size, 48)
        # actions => tensorbatch_size, 4)
        # Q_expected => tensor(batch_size 1)
        Q_expected = self.critic_local(states, actions)

        # Compute critic loss
        if use_huber_loss:
            critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()

        if grad_clip_critic is not None:
            clip_grad_norm_(self.critic_local.parameters(),
                            grad_clip_critic)

        self.critic_optim.step()

        soft_update(self.critic_local, self.critic_target, tau)

    def update_actor(self, states, pred_actions):
        grad_clip_actor = self.config.grad_clip_actor
        tau = self.config.tau

        actor_loss = -self.critic_local(states, pred_actions).mean()

        # Minimize the loss
        self.actor_optim.zero_grad()
        actor_loss.backward()

        if grad_clip_actor is not None:
            clip_grad_norm_(self.actor_local.parameters(),
                            grad_clip_actor)

        self.actor_optim.step()

        soft_update(self.actor_local, self.actor_target, tau)

    def train(self):
        num_episodes = self.config.num_episodes
        max_steps = self.config.max_steps
        log_every = self.config.log_every
        env_solved = self.config.env_solved
        times_solved = self.config.times_solved
        env = self.config.env

        start = time.time()

        scores_window = deque(maxlen=times_solved)
        best_score = -np.inf
        scores = []

        for i_episode in range(1, num_episodes+1):
            state = self.reset()
            score = 0

            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                self.step(state, action, reward, next_state, done)

                state = next_state
                score += reward

                if done: break

            scores.append(score)
            scores_window.append(score)
            avg_score = np.mean(scores_window)

            print('\rEpisode {}\tAvg Score: {:.3f}'
                  .format(i_episode, avg_score), end='')

            if i_episode % log_every == 0:
                print('\rEpisode {}\tAvg Score: {:.3f}'
                      .format(i_episode, avg_score))

            if avg_score > best_score:
                best_score = avg_score
                self.save_weights()

            if avg_score >= env_solved:
                print('\nRunning evaluation without noise...')

                avg_score = self.eval_episode()

                if avg_score >= env_solved:
                    time_elapsed = get_time_elapsed(start)

                    print('Environment solved {} times consecutively!'.format(times_solved))
                    print('Avg score: {:.3f}'.format(avg_score))
                    print('Time elapsed: {}'.format(time_elapsed))
                    break
                else:
                    print('No success. Avg score: {:.3f}'.format(avg_score))

        env.close()

        return scores

    def save_weights(self):
        torch.save(self.actor_local.state_dict(),
                   '{}_actor_checkpoint.ph'.format(self.name))

    def load_weights(self):
        self.actor_local.\
            load_state_dict(
                torch.load('{}_actor_checkpoint.ph'.
                           format(self.name),
                           map_location='cpu'))

    def eval_episode(self):
        times_solved = self.config.times_solved
        env = self.config.env
        
        total_reward = 0
        
        for _ in range(times_solved):
            state = env.reset()
            while True:
                actions = self.act(state, train=False)
                state, reward, done, _ = env.step(actions)

                total_reward += reward
    
                if done: break
                
        return total_reward / times_solved

    def summary(self, agent_name='DDGP Agent'):
        print('{}:'.format(agent_name))
        print('==========')
        print('')
        print('Actor Network:')
        print('--------------')
        print(self.actor_local)
        print('')
        print('Critic Network:')
        print('---------------')
        print(self.critic_local)