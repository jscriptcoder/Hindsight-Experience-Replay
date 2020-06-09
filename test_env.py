import gym
import time
import gym.spaces
import rocket_lander_gym

PRINT_DEBUG_MSG = True

# RocketLander-v0 | LunarLanderContinuous-v2 | MountainCar-v0 | CartPole-v0
env = gym.make('LunarLanderContinuous-v2')
# env = HerEnvWrapper(env)
env.reset()

step = 0
while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    step += 1

    if PRINT_DEBUG_MSG:
        print("Step          ", step)
        print("Action Taken  ", action)
        print("Observation   ", observation)
        print("Reward Gained ", reward)
        print("Info          ", info, end='\n\n')

    if done:
        print("Simulation done.")
        break

env.close()
