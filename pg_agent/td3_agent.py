import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .ddpg_agent import DDPGAgent
from .critic import Critic
from .utils import soft_update
from .device import device

class TD3Agent(DDPGAgent):
    """Twin Delayed DDPG agents

    See for more info:
        https://spinningup.openai.com/en/latest/algorithms/td3.html
        https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
    """

    name = 'TD3'

    def __init__(self, config):
        super().__init__(config)

        self.twin_local = Critic(config.state_size,
                                 config.action_size,
                                 config.hidden_critic,
                                 config.activ_critic)

        self.twin_target = Critic(config.state_size,
                                  config.action_size,
                                  config.hidden_critic,
                                  config.activ_critic)

        self.twin_target.load_state_dict(self.twin_local.state_dict())

        self.twin_optim = config.optim_critic(self.twin_local.parameters(),
                                              lr=config.lr_critic, 
                                              weight_decay=config.critic_weight_decay)

    def update_critic(self,
                      states,
                      actions,
                      next_states,
                      next_actions,
                      rewards,
                      dones):

        policy_noise = self.config.policy_noise
        noise_clip = self.config.noise_clip
        grad_clip_critic = self.config.grad_clip_critic
        use_huber_loss = self.config.use_huber_loss
        gamma = self.config.gamma
        tau = self.config.tau

        # Target Policy Smoothing Regularization: add a small amount of clipped
        # random noises to the selected action
        if policy_noise > 0.0:
            noise = torch.normal(torch.zeros(next_actions.size()),
                                 policy_noise).to(device)
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            next_actions = (next_actions + noise).clamp(-1., 1.)

        Q_targets_next1 = \
            self.critic_target(next_states, next_actions)
        
        Q_targets_next2 = \
            self.twin_target(next_states, next_actions)

        Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2).detach()

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected1 = self.critic_local(states, actions)
        Q_expected2 = self.twin_local(states, actions)

        # Compute critic loss
        if use_huber_loss:
            critic_loss = F.smooth_l1_loss(Q_expected1, Q_targets)
            twin_loss = F.smooth_l1_loss(Q_expected2, Q_targets)
        else:
            critic_loss = F.mse_loss(Q_expected1, Q_targets)
            twin_loss = F.mse_loss(Q_expected2, Q_targets)
        
        self.value_losses.append(torch.max(critic_loss, twin_loss).item())

        # Minimize the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()

        if grad_clip_critic is not None:
            clip_grad_norm_(self.critic_local.parameters(), grad_clip_critic)

        self.critic_optim.step()

        self.twin_optim.zero_grad()
        twin_loss.backward()

        if grad_clip_critic is not None:
            clip_grad_norm_(self.twin_local.parameters(), grad_clip_critic)

        self.twin_optim.step()

    def update_target_networks(self):
        tau = self.config.tau
        soft_update(self.critic_local, self.critic_target, tau)
        soft_update(self.twin_local, self.twin_target, tau)
        soft_update(self.actor_local, self.actor_target, tau)

    def summary(self, agent_name='TD3 Agent'):
        super().summary(agent_name)
        print('')
        print('Twin Network:')
        print('-------------')
        print(self.twin_local)