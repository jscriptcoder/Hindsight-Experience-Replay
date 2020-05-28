import time
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent
from .actor import Actor
from .critic import Critic
from .noise import OUNoise, GaussianNoise
from .utils import soft_update, make_experience, from_experience, get_time_elapsed
from .device import device


class DDPGAgent(Agent):
    """Deep Deterministic Policy Gradient
    https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
    """

    name = 'DDPG'

    def __init__(self, config):
        super().__init__(config)
                
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(config.state_size,
                                 config.action_size,
                                 config.hidden_actor,
                                 config.activ_actor)

        self.actor_target = Actor(config.state_size,
                                  config.action_size,
                                  config.hidden_actor,
                                  config.activ_actor)

        self.actor_target.load_state_dict(self.actor_local.state_dict())

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

        self.critic_target.load_state_dict(self.critic_local.state_dict())

        self.critic_optim = config.optim_critic(self.critic_local.parameters(), 
                                                lr=config.lr_critic, 
                                                weight_decay=config.critic_weight_decay)

        # Noise process
        if config.use_ou_noise:
            self.noise = OUNoise(config.action_size,
                                 config.ou_mu,
                                 config.ou_theta,
                                 config.ou_sigma)
        else:
            self.noise = GaussianNoise(config.action_size, 
                                       config.expl_noise)

        self.noise_weight = config.noise_weight

    def act(self, state, train=True):
        decay_noise = self.config.decay_noise
        use_linear_decay = self.config.use_linear_decay
        noise_linear_decay = self.config.noise_linear_decay
        noise_decay = self.config.noise_decay
        noise_weight_min = self.config.noise_weight_min

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if train:
            action += self.noise.sample() * self.noise_weight

            if decay_noise:
                if use_linear_decay:
                    self.noise_weight = max(noise_weight_min,
                                            self.noise_weight - noise_linear_decay)
                else:
                    self.noise_weight = max(noise_weight_min,
                                            self.noise_weight * noise_decay)

        return np.clip(action, -1., 1.)

    def reset(self):
        self.noise.reset()
        return super().reset()

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

        Q_targets_next = \
            self.critic_target(next_states, next_actions).detach()

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        if use_huber_loss:
            critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.value_losses.append(critic_loss.item())

        # Minimize the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()

        if grad_clip_critic is not None:
            clip_grad_norm_(self.critic_local.parameters(),
                            grad_clip_critic)

        self.critic_optim.step()

    def update_actor(self, states, pred_actions):
        grad_clip_actor = self.config.grad_clip_actor

        actor_loss = -self.critic_local(states, pred_actions).mean()

        self.policy_losses.append(actor_loss.item())

        # Minimize the loss
        self.actor_optim.zero_grad()
        actor_loss.backward()

        if grad_clip_actor is not None:
            clip_grad_norm_(self.actor_local.parameters(),
                            grad_clip_actor)

        self.actor_optim.step()

    def update_target_networks(self):
        tau = self.config.tau
        soft_update(self.critic_local, self.critic_target, tau)
        soft_update(self.actor_local, self.actor_target, tau)

    def learn(self, experiences):
        policy_freq_update = self.config.policy_freq_update

        (states,
         actions,
         rewards,
         next_states,
         dones) = from_experience(experiences)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target(next_states)

        self.update_critic(states,
                           actions,
                           next_states,
                           next_actions,
                           rewards,
                           dones)

        self.p_update = (self.p_update + 1) % policy_freq_update

        if self.p_update == 0:
            # ---------------------------- update actor ---------------------------- #
            pred_actions = self.actor_local(states)
            
            self.update_actor(states, pred_actions)

            self.update_target_networks()

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
