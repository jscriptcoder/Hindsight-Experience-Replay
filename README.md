# Hindsight Experience Replay

Controlling a Spaceship using Hindsight Experience Replay (a.k.a HER)

This research is based on the paper [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) submitted on Jul 5th, 2017 by OpenAI Researchers

I wrote a [Medium article]() trying to demystify this algorithm, where I describe my journey during the reaserch.

## Abstract

Dealing with sparse rewards is one of the biggest challenges in Reinforcement Learning (RL). We present a novel technique called Hindsight Experience Replay which allows sample-efficient learning from rewards which are sparse and binary and therefore avoid the need for complicated reward engineering. It can be combined with an arbitrary off-policy RL algorithm and may be seen as a form of implicit curriculum.

We demonstrate our approach on the task of manipulating objects with a robotic arm. In particular, we run experiments on three different tasks: pushing, sliding, and pick-and-place, in each case using only binary rewards indicating whether or not the task is completed. Our ablation studies show that Hindsight Experience Replay is a crucial ingredient which makes training possible in these challenging environments. We show that our policies trained on a physics simulation can be deployed on a physical robot and successfully complete the task.

## Resources

#### Papers

1. [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
2. [DHER: Hindsight Experience Replay for Dynamic Goals](https://openreview.net/forum?id=Byf5-30qFX)
3. [Hindsight policy gradients](https://arxiv.org/abs/1711.06006)
4. [Rewriting History with Inverse RL: Hindsight Inference for Policy Improvement](https://arxiv.org/abs/2002.11089)
5. [Advances in Experience Replay](https://arxiv.org/abs/1805.05536)
6. [Curriculum-guided Hindsight Experience Replay](https://papers.nips.cc/paper/9425-curriculum-guided-hindsight-experience-replay)
7. [Soft Hindsight Experience Replay](https://arxiv.org/abs/2002.02089)

#### Articles

1. [Reinforcement Learning with Hindsight Experience Replay](https://towardsdatascience.com/reinforcement-learning-with-hindsight-experience-replay-1fee5704f2f8)
2. [Learning from mistakes with Hindsight Experience Replay](https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305)
3. [Advanced Exploration: Hindsight Experience Replay](https://medium.com/analytics-vidhya/advanced-exploration-hindsight-experience-replay-fd604be0fc4a)
4. [Understanding DQN+HER](https://deeprobotics.wordpress.com/2018/03/07/bitflipper-herdqn/)

#### Videos

1. [Hindsight Experience Replay | Two Minute Papers #192](https://www.youtube.com/watch?v=Dvd1jQe3pq0)
2. [Overcoming sparse rewards in Deep RL: Curiosity, hindsight & auxiliary tasks](https://www.youtube.com/watch?v=0Ey02HT_1Ho)
3. [Hindsight Experience Replay](https://www.youtube.com/watch?v=Dz_HuzgMxzo)
4. [Hindsight Experience Replay by Olivier Sigaud](https://www.youtube.com/watch?v=77xkqEAsHFI)

#### Repos

1. [2D Gridworld navigation using RL with Hindsight Experience Replay](https://github.com/orrivlin/Navigation-HER)
2. [Pytorch implementation of Hindsight Experience Replay (HER)](https://github.com/TianhongDai/hindsight-experience-replay)
3. [Reproducing results from the Hindsight Experience Replay paper in PyTorch](https://github.com/viraat/hindsight-experience-replay)
4. [Hindsight Experience Replay by Alex Hermansson](https://github.com/AlexHermansson/hindsight-experience-replay)
