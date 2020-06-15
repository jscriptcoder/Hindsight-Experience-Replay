import time
import random
import numpy as np
import torch
from collections import deque

from .replay_buffer import ReplayBuffer
from .utils import make_experience, get_time_elapsed, goal_conditioned_reward

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
            # Sample a goal g and an initial state s0
            state = self.reset()
            goal = np.array([0., 0., 0., 0., 0., 0., 1., 1.]) # LunarLander goal
            score = 0

            episode = []
            additional_goals = []

            for time_step in range(max_steps):
                action = self.act(np.concatenate((state, goal)))

                next_state, original_reward, done, _ = env.step(action)

                episode.append((state, action, next_state, done))
                additional_goals.append(next_state)

                state = next_state
                score += original_reward

                if done: break

            for state, action, next_state, done in episode:
                
                # Sparse reward
                reward = goal_conditioned_reward(next_state, goal)

                transition = make_experience(np.concatenate((state, goal)),
                                            action,
                                            reward,
                                            np.concatenate((next_state, goal)),
                                            done)
                self.memory.add(transition)

                goals = random.sample(additional_goals, k=8)

                for additional_goal in goals:
                    reward = goal_conditioned_reward(next_state, additional_goal)
                    # done = True if reward == 1 else done

                    transition = make_experience(np.concatenate((state, additional_goal)),
                                                action,
                                                reward,
                                                np.concatenate((next_state, additional_goal)),
                                                done)
                    self.memory.add(transition)
                
            self.sample_and_learn()

            scores.append(score)
            scores_window.append(score)
            avg_score = np.mean(scores_window)
            # avg_policy_loss = np.mean(self.policy_losses)
            # avg_value_loss = np.mean(self.value_losses)
            
            # to_print = '\rEpisode {}\tScore: {:5.2f}\tAvg Score: {:5.2f}\tAvg Policy Loss: {:5.2f}\tAvg Value Loss: {:5.2f}'\
            #             .format(i_episode, score, avg_score, avg_policy_loss, avg_value_loss)

            to_print = '\rEpisode {}\tScore: {:5.2f}\tAvg Score: {:5.2f}'\
                        .format(i_episode, score, avg_score)

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

    def summary(self, agent_name='Agent'):
        pass
