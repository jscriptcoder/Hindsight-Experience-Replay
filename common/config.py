from torch.nn import ReLU
from torch.optim import Adam

class Config():
    seed = 0
    env = None

    # When we reach env_solved avarage score (our target score for this environment),
    # we'll run a full evaluation, that means, we're gonna evaluate times_solved #
    # times (this is required to solve the env) and avarage all the rewards:
    env_solved = 1
    times_solved = 100

    buffer_size = int(1e6)
    batch_size = 128
    num_episodes = 2000
    num_updates = 2 # how many updates we want to perform in one learning step
    max_steps = 2000 # max steps done per episode if done is never True
    
    # Reward after reaching `max_steps` (punishment, hence negative reward)
    max_steps_reward = None
    
    state_size = None
    action_size = None
    action_limit = 1. # range continuous action [-action_limit, +action_limit]
    gamma = 0.99 # discount factor
    tau = 1e-3 # interpolation param, used in polyak averaging (soft update)
    lr_actor = 3e-4
    lr_critic = 3e-4
    hidden_actor = (256, 256)
    hidden_critic = (256, 256)
    activ_actor = ReLU()
    activ_critic = ReLU()
    optim_actor = Adam
    optim_critic = Adam
    critic_weight_decay = 1e-2 # L2 weight decay
    grad_clip_actor = None # gradient clipping for actor network
    grad_clip_critic = None # gradient clipping for critic network
    use_huber_loss = False # whether to use huber loss (True) or mse loss (False)
    update_every = 4 # how many steps before updating networks

    use_ou_noise = True # whether to use OU (True) or Gaussian (False) noise
    ou_mu = 0.0
    ou_theta = 0.15
    ou_sigma = 0.2
    expl_noise = 0.1 # exploration noise in case of using Gaussian
    noise_weight = 1.0
    noise_weight_min = 0.1
    decay_noise = False
    
    # noise_weight - noise_linear_decay (True), noise_weight * noise_decay (False)
    use_linear_decay = False

    noise_linear_decay = 1e-6
    noise_decay = 0.99

    log_every = 100
    policy_noise = 0.2 # target policy smoothing by adding noise to the target action
    noise_clip = 0.5 # clipping value for the noise added to the target action
    policy_freq_update = 2 # how many critic net updates before updating the actor

    log_std_min=-20 # min value of the log std calculated by the Gaussian policy
    log_std_max=2 # max value of the log std calculated by the Gaussian policy
    alpha = 0.01
    alpha_auto_tuning = True # when True, alpha is a learnable
    optim_alpha = Adam # optimizer for alpha
    lr_alpha = 3e-4 # learning rate for alpha