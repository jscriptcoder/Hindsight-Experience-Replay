import copy
import random
import numpy as np
import warnings
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from common.replay_buffer import ReplayBuffer
from common.utils import make_experience, from_experience, seed_all, sample_transitions
from common.device import device
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

        self.soft_update(1.0)

        self.optimizer = optim.Adam(self.qn_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.losses = deque(maxlen=100)
        self.t_step = 0

    def process_input(self, state, goal=None):
        use_her = self.config.use_her

        if use_her and goal is not None:
            state = np.concatenate([state, goal])

        return state

    def act(self, state, goal, eps=0.0, use_target=False):
        action_size = self.config.action_size

        state_input = self.process_input(state, goal)
        state_input = torch.from_numpy(state_input).float().unsqueeze(0).to(device)

        model = self.qn_target if use_target else self.qn_local

        model.eval()
        with torch.no_grad():
            action_values = model(state_input)
        model.train()

        # ε-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().numpy())
        else:
            return random.choice(np.arange(action_size))

    def step(self, state, action, reward, next_state, done, goal):
        update_every = self.config.update_every
        batch_size = self.config.batch_size

        self.add_experience(state, action, reward, next_state, done, goal)

        self.t_step = (self.t_step + 1) % update_every

        if self.t_step == 0:
            if len(self.memory) > batch_size:
                experiences = self.memory.sample()
                return self.learn(experiences)

    def learn(self, experiences):
        (states, actions, rewards, next_states, dones) = from_experience(experiences)

        tau = self.config.tau
        gamma = self.config.gamma
        use_her = self.config.use_her
        use_double = self.config.use_double
        use_huber_loss = self.config.use_huber_loss

        if use_double:
            # Double DQN: https://arxiv.org/abs/1509.06461
            best_action = self.qn_local(next_states).argmax(-1, keepdim=True)
            max_q = self.qn_target(next_states).detach().gather(-1, best_action)
        else:
            max_q = self.qn_target(next_states).detach().max(-1, keepdim=True)[0]

        q_targets = rewards + (gamma * max_q * (1 - dones))

        # Clipping targets as suggested in the paper: 
        # See A Experiment details / Training procedure
        if use_her:
            clip_return = 1 / (1 - gamma)
            q_targets = torch.clamp(q_targets, -clip_return, clip_return)

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
        for target_param, local_param in zip(
            self.qn_target.parameters(), 
            self.qn_local.parameters()
        ):
            target_param.data.copy_(tau * local_param + (1.0 - tau) * target_param)

    def add_experience(self, state, action, reward, next_state, done, goal=None):
        use_her = self.config.use_her

        state_ = self.process_input(state, goal)
        next_state_ = self.process_input(next_state, goal)

        experience = make_experience(state_, action, reward, next_state_, done, {})
        self.memory.add(experience)

    def eval_episode(self, env, use_target=False):

        times_eval = self.config.times_eval
        max_steps = self.config.max_steps

        total_reward = 0

        for _ in range(times_eval):
            state, goal = env.reset()
            done = False

            while not done:
                action = self.act(state, goal, use_target=use_target)
                state, reward, done, info = env.step(action)

                total_reward += reward

        return total_reward / times_eval

    def train(self, env):
        print("Training on {}".format(device))

        eps = self.config.eps_start
        eps_end = self.config.eps_end
        eps_decay = self.config.eps_decay
        episodes = self.config.episodes
        max_steps = self.config.max_steps
        dist_tolerance = self.config.dist_tolerance
        dense_reward = self.config.dense_reward
        use_her = self.config.use_her
        future_k = self.config.future_k
        eval_every = self.config.eval_every

        writer = SummaryWriter(comment="_LunarLander")

        most_reached = 1
        best_score = -np.inf

        ### DQN algorithm ###
        for episode in range(episodes):

            total_reward = 0
            target_reached = 0
            trajectory = []
            state, goal = env.reset()

            for _ in range(max_steps):
                # With probability eps select a random action, otherwise select max
                action = self.act(state, goal, eps)

                # Execute action and observe next state and reward
                next_state, reward, done, info = env.step(action)

                # Store transition and perform optimization step
                self.step(state, action, reward, next_state, done, goal)

                # Store for potential use in HER
                trajectory.append(
                    make_experience(state, 
                                    action, 
                                    reward, 
                                    next_state, 
                                    done, 
                                    info)
                )

                total_reward += reward
                target_reached += 1 if info["success"] else 0
                state = next_state

                if done:
                    break
            
            ### HER does her magic here ###
            if use_her:
                steps_taken = len(trajectory)
                
                # Replay transitions with different goals
                for t in range(steps_taken):
                    state, action, _, next_state, _, info = copy.deepcopy(trajectory[t])
                    
                    # Convert next state into a goal
                    achieved_goal = env.to_goal(next_state)

                    # Will sample final or future random transitions depending on 'future_k'
                    #   future_k = 1 => final strategy
                    #   future_k > 1 => future strategy
                    selected_transitions = sample_transitions(trajectory, t, future_k)

                    # Loop over virtual goals. These are achieved goals along the episode
                    for transition in selected_transitions:
                        additional_goal = env.to_goal(transition.state)

                        # Recompute reward
                        reward, _ = env.compute_reward(achieved_goal, 
                                                       additional_goal, 
                                                       eps=dist_tolerance, 
                                                       dense=dense_reward)

                        # Store in buffer, with a new goal, and perform optimization step
                        self.step(state, 
                                  action, 
                                  reward, 
                                  next_state, 
                                  False, # we're not done even if goal is reached (keep hovering)
                                  additional_goal)
            ### End HER ###

            avg_loss = np.mean(self.losses)

            print(
                "episode: {}, exploration: {:.2f}%, target reached: {}, total reward: {:.2f}, avg Loss: {:.3f}".format(
                    episode + 1, 
                    100 * eps, 
                    target_reached, 
                    total_reward, 
                    avg_loss
                )
            )
            
            writer.add_scalar("Target Reached", target_reached, episode)
            writer.add_scalar("Episode Reward", total_reward, episode)
            writer.add_scalar("Avg Loss", avg_loss, episode)

            if (episode+1) % eval_every == 0:
                print("\nRunning evaluation...")

                score = self.eval_episode(env, use_target=False)

                print("eval score: {:.2f}".format(score))
                writer.add_scalar("Evaluation Score", score, episode)

                if score > best_score:
                    best_score = score
                    torch.save(self.qn_local.state_dict(), "best_weights.pth")
            
            eps = max(eps_end, eps_decay * eps)

        ### End DQN ###

