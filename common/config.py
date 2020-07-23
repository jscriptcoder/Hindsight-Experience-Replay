class Config:
    buffer_size = int(1e6)
    batch_size = 128
    gamma = 0.99
    tau = 1e-3
    episodes = 2000
    max_steps = 1000
    future_k = 4
    state_size = None
    action_size = None
    goal_size = None
    lr = 1e-3
    eps_start = 1.0
    eps_end = 0.1
    eps_decay = 0.995
    eval_every = 10
    use_her = True
    use_double = False
    use_huber_loss = False