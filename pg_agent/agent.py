import time
import random
import numpy as np
import torch
from collections import deque

from common.replay_buffer import ReplayBuffer
from common.utils import make_experience, get_time_elapsed
from common.utils import get_reward, random_sample

class Agent():
    """Common logic"""

    name = 'Agent'

    def __init__(self, config):
        self.config = config

        self.memory = ReplayBuffer(config.buffer_size, config.batch_size)

        self.t_step = 0
        self.p_update = 0
        
        self.policy_losses = []
        self.value_losses = []

    def act(self, state, train=True):
        pass

    def reset(self):
        self.policy_losses = []
        self.value_losses = []
        return self.config.env.reset()

    def update_target_networks(self):
        pass

    def learn(self, experiences):
        pass

    def sample_and_learn(self):
        batch_size = self.config.batch_size
        update_every = self.config.update_every
        num_updates = self.config.num_updates

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % update_every

        if self.t_step == 0:

            # Learn, if enough samples are available in memory
            if len(self.memory) > batch_size:

                # Multiple updates in one learning step
                for _ in range(num_updates):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def step(self, state, action, reward, next_state, done):
        experience = make_experience(state,
                                     action,
                                     reward,
                                     next_state,
                                     done)
        self.memory.add(experience)

        self.sample_and_learn()

    def train(self):
        num_episodes = self.config.num_episodes
        max_steps = self.config.max_steps
        max_steps_reward = self.config.max_steps_reward
        log_every = self.config.log_every
        env_solved = self.config.env_solved
        times_solved = self.config.times_solved
        env = self.config.env

        start = time.time()

        # writer = SummaryWriter()
        scores_window = deque(maxlen=times_solved)
        best_score = -np.inf
        scores = []

        for i_episode in range(1, num_episodes+1):
            state = self.reset()
            score = 0

            for time_step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                self.step(state, action, reward, next_state, done)
                
                if not done and time_step == max_steps-1:
                    # We reached max_steps
                    done = True
                    
                    # Do we penalized?
                    reward = max_steps_reward if max_steps_reward is not None else reward

                state = next_state
                score += reward
                
                if done: break

            scores.append(score)
            scores_window.append(score)
            avg_score = np.mean(scores_window)
            avg_policy_loss = np.mean(self.policy_losses)
            avg_value_loss = np.mean(self.value_losses)
            
            to_print = '\rEpisode {}\tScore: {:5.2f}\tAvg Score: {:5.2f}\tAvg Policy Loss: {:5.2f}\tAvg Value Loss: {:5.2f}'\
                        .format(i_episode, score, avg_score, avg_policy_loss, avg_value_loss)

            print(to_print, end='')

            if i_episode % log_every == 0: print(to_print)

            if avg_score > best_score:
                best_score = avg_score
                self.save_weights()

            if avg_score >= env_solved:
                print('\nRunning evaluation...')

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

    def train_her(self):
        num_episodes = self.config.num_episodes
        max_steps = self.config.max_steps
        max_steps_reward = self.config.max_steps_reward
        log_every = self.config.log_every
        env_solved = self.config.env_solved
        times_solved = self.config.times_solved
        env = self.config.env

        start = time.time()

        # writer = SummaryWriter()
        scores_window = deque(maxlen=times_solved)
        best_score = -np.inf
        scores = []

        for i_episode in range(1, num_episodes+1):
            # Sample a goal g and an initial state s0.
            state = self.reset()
            goal = np.array([0., 0., 0., 0., 0., 0., 1., 1.])

            score = 0

            episode = []
            achieved_goals = []

            for time_step in range(max_steps):
                # Sample an action at using the behavioral policy from A:
                # at ← πb(st||g)
                action = self.act(np.concatenate((state, goal)))

                # Execute the action at and...
                next_state, original_reward, done, _ = env.step(action)

                #  observe a new state st+1
                episode.append((state, action, next_state, done))

                achieved_goals.append(next_state)

                state = next_state
                score += original_reward

                if done: break

            for i, (state, action, next_state, done) in enumerate(episode):
                
                # rt := r(st, at, g)
                reward = get_reward(next_state, goal)

                # Store the transition (st||g, at, rt, st+1||g) in R
                transition = make_experience(np.concatenate((state, goal)),
                                            action,
                                            reward,
                                            np.concatenate((next_state, goal)),
                                            done)
                self.memory.add(transition)

                # Sample a set of additional goals for replay G := S(current episode)
                # additional_goals = random_sample(achieved_goals[i:]) # future strategy

                # for additional_goal in additional_goals:
                #     # r' := r(st, at, g')
                #     reward = get_reward(next_loc, additional_goal)
                    
                #     # Store the transition (st||g', at, rt, st+1||g') in R
                #     transition = make_experience(np.concatenate((obs, additional_goal)),
                #                                 action,
                #                                 reward,
                #                                 np.concatenate((next_obs, additional_goal)),
                #                                 done)

                #     self.memory.add(transition)

                additional_goal = achieved_goals[-1] # m(st)

                # r' := r(st, at, g')
                reward = get_reward(next_state, additional_goal)
                
                # Store the transition (st||g', at, rt, st+1||g') in R
                transition = make_experience(np.concatenate((state, additional_goal)),
                                            action,
                                            reward,
                                            np.concatenate((next_state, additional_goal)),
                                            done)

                self.memory.add(transition)
            
            # for t = 1, N do
            # Sample a minibatch B from the replay buffer R
            # Perform one step of optimization using A and minibatch B
            self.sample_and_learn()
            # end for
            
            scores.append(score)
            scores_window.append(score)
            avg_score = np.mean(scores_window)
            avg_policy_loss = np.mean(self.policy_losses)
            avg_value_loss = np.mean(self.value_losses)
            
            to_print = '\rEpisode {}\tScore: {:5.2f}\tAvg Score: {:5.2f}\tAvg Policy Loss: {:5.2f}\tAvg Value Loss: {:5.2f}'\
                        .format(i_episode, score, avg_score, avg_policy_loss, avg_value_loss)

            # to_print = '\rEpisode {}\tScore: {:5.2f}\tAvg Score: {:5.2f}'\
            #             .format(i_episode, score, avg_score)

            print(to_print, end='')

            if i_episode % log_every == 0: 
                print(to_print)

                print('\nRunning evaluation...')

                avg_score = self.eval_episode_her()

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

    def eval_episode_her(self):
        times_solved = self.config.times_solved
        env = self.config.env
        
        total_reward = 0
        
        for _ in range(times_solved):
            state = env.reset()
            goal = np.array([0., 0., 0., 0., 0., 0., 1., 1.])

            while True:
                actions = self.act(np.concatenate((state, goal)), train=False)
                state, reward, done, _ = env.step(actions)

                total_reward += reward
    
                if done: break
                
        return total_reward / times_solved

    def summary(self, agent_name='Agent'):
        pass
