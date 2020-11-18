# Implamentation-and-Some-Modification-about-DQN-and-SAC
In this project, I explore some typical value-based and policy-based RL algorithms. I do experiments on DQN and its six variants and their combination in Atari environments Pong and Boxing. I also do some experiments on SAC with DDPG as baseline on three MuJoCo environments Hopper-v2, Ant-v2, and HalfCheetah-v2. Implamentation and modification details can be found in [Here](FinalProjectReport.pdf) 




## Policy-based RL

Pytorch SAC and DDPG implementation for MuJoCo environments.

I've tried Hopper-v2, Ant-v2, and HalfCheetah-v2 and got similar results to SAC paper. 

### Content

* SAC (fixed temperature&learned temperature)
* DDPG
* SAC(noisy net & multi-step). This doesn't work well.

### Dependencies

* python 3.6
* numpy
* pytorch 1.1.0
* tensorboard 2.0.0
* MuJoCo & mujoco-py

#### Usage

For SAC with fixed temperature (alpha=0.2 works well for Hopper, Ant, and HalfCheetah. Here I take Ant as example.)

```
python run.py --env-name Ant-v2 --alpha 0.2
```

For SAC with learned temperature

```
python run.py --env-name Ant-v2 --automatic_entropy_tuning True
```

For DDPG

```
python run.py --env-name Ant-v2 --policy Deterministic
```

The commands above use soft update. If you want to use hard update(for SAC or DDPG), you need to set `tau` to 1 and set `target_update_interval`. For example,

```
python run.py --env-name Ant-v2 --alpha 0.2 --tau 1 --target_update_interval 1000
```



### Arguments

#### Basic Arguments

| Name                       | Meaning                                           | Default        |
| -------------------------- | ------------------------------------------------- | -------------- |
| --env-name                 | Mujoco Gym environment                            | HalfCheetah-v2 |
| --seed                     | Random seed                                       | 123456         |
| --logdir                   | The folder that store log info                    | runs/          |
| --policy                   | Policy Type: Gaussian(SAC) \| Deterministic(DDPG) | Gaussian       |
| --noisy                    | Whether add noise to  Q network                   | False          |
| --multi-step               | N-Step Learning                                   | 1              |
| --automatic_entropy_tuning | Automatically adjust temperature                  | False          |
| --eval                     | Whether test policy during training               | True           |



#### Training Arguments

| Name                     | Meaning                                               | Default |
| ------------------------ | ----------------------------------------------------- | ------- |
| --gamma                  | Discount factor                                       | 0.99    |
| --tau                    | target smoothing coefficient(τ)                       | 0.005   |
| --lr                     | learning rate                                         | 0.0003  |
| --alpha                  | Temperature parameter                                 | 0.2     |
| --batch_size             | Batch size                                            | 256     |
| --num_steps              | maximum number of steps                               | 3000001 |
| --hidden_size            | Hidden unite size                                     | 256     |
| --updates_per_step       | Number of steps between optimization step             | 1       |
| --start_steps            | How many transitions collected before learning starts | 10000   |
| --target_update_interval | Interval of target network update                     | 1       |
| --replay_size            | Size of replay buffer                                 | 1000000 |
| --sigma-init             | Sigma initialization value for NoisyNet               | 0.4     |



### Reference

 [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf).

[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) 

[Learning to Walk via Deep Reinforcement Learning](https://arxiv.org/pdf/1812.11103.pdf)

[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)






## DQN

Pytorch DQN and its variants implementation to play Pong and Boxing.

In theory, this code can work on other Atari environments that return images as observations, but re-tuning hyperparameters may be necessary. 

### Content

-  Nature DQN 
-  Double DQN
-  Prioritized Experience Replay 
-  Dueling DQN
-  Multi-step DQN
-  Distributional DQN
-  Noisy DQN

### Dependencies

* python 3.6
* numpy
* pytorch 1.1.0
* tensorboard 2.0.0
* opencv-python `pip install opencv-python`
* gym `pip install gym gym[atari]`

### Usage

Train nature DQN with the default arguments on gym Atari environments:

```
python run.py --env-name BoxingNoFrameskip-v4
```

Train combined DQN with tuned hyperparameters for PongNoFrameskip-v4, which performs best:

```
python run.py --tuned-pong --double --dueling --noisy --prioritized-replay --multi-step 2
```

Train combined DQN with tuned hyperparameters for BoxingNoFrameskip-v4, which performs best:

```
python run.py --tuned-boxing --double --dueling --noisy --prioritized-replay --multi-step 3
```



### Important Arguments

#### Algorithm Arguments

| Name                 | Meaning                               | Default           |
| -------------------- | ------------------------------------- | ----------------- |
| --seed               | Random seed                           | 1112              |
| --tuned-pong         | Use tuned hyper-parameters for Pong   | False(store_true) |
| --tuned-boxing       | Use tuned hyper-parameters for Boxing | False(store_true) |
| --double             | Enable Double-Q Learning              | False(store_true) |
| --dueling            | Enable Dueling Network                | False(store_true) |
| --noisy              | Enable Noisy Network                  | False(store_true) |
| --prioritized-replay | Enable prioritized experience replay  | False(store_true) |
| --distributional     | Enable categorical QDN                | False(store_true) |
| --multi-step         | N-Step Learning                       | 1                 |

There are some parameters for above algorithms which can be found in code run.py. Here store_true means using `--double` means true and we don't need to use `--double True`.

#### Training Arguments

| Name                  | Meaning                                               | Default |
| --------------------- | ----------------------------------------------------- | ------- |
| --batch-size          | Batch size                                            | 32      |
| --buffer-size         | Maximum memory buffer size                            | 100000  |
| --update-target       | Interval of target network update                     | 1000    |
| --max-frames          | Number of frames to train                             | 1500000 |
| --train-freq          | Number of steps between optimization step             | 1       |
| --gamma               | Discount factor                                       | 0.99    |
| --learning-start      | How many transitions collected before learning starts | 10000   |
| --eps-start           | Start value of epsilon                                | 1       |
| --eps-final           | Final value of epsilon                                | 0.02    |
| --eps-decay           | How many steps it takes from eps-start to eps-final   | 100000  |
| --soft-update         | Whether use soft update                               | False   |
| --tau                 | Target smoothing coefficient(τ)                       | 0.005   |
| --max-eps-step        | Max steps of an episode                               | 27000   |
| --lr                  | Learning rate                                         | 1e-4    |
| --evaluation-interval | Frames for evaluation interval                        | 10000   |
| --eval-time           | How many episode to eval                              | 10      |
| --logdir              | the folder that store log info                        | runs/   |

#### Environment Arguments

| Name           | Meaning                                   | Default            |
| -------------- | ----------------------------------------- | ------------------ |
| --env          | Environment Name                          | PongNoFrameskip-v4 |
| --episode-life | Whether env has episode life(1) or not(0) | 1                  |
| --clip-rewards | Whether env clip rewards(1) or not(0)')   | 1                  |
| --frame-stack  | Whether env stacks frame(1) or not(0)     | 1                  |
| --scale        | Whether env scales(1) or not(0)           | 0                  |

#### Test Arguments

| Name         | Meaning                                    | Dafult              |
| ------------ | ------------------------------------------ | ------------------- |
| --load-model | Pretrained model name to load (state dict) | None                |
| --evaluate   | Whether test only                          | False('store_true') |
| --render     | Whether render when testing                | False('store_true') |



### Related Papers

1. [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518 (7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
2. [van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv preprint arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)
3. [T. Schaul et al., "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
4. [Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning." arXiv preprint arXiv:1511.06581, 2015.](https://arxiv.org/pdf/1511.06581.pdf)
5. [M. Fortunato et al., "Noisy Networks for Exploration." arXiv preprint arXiv:1706.10295, 2017.](https://arxiv.org/pdf/1706.10295.pdf)
6. [M. G. Bellemare et al., "A Distributional Perspective on Reinforcement Learning." arXiv preprint arXiv:1707.06887, 2017.](https://arxiv.org/pdf/1707.06887.pdf)
7. [R. S. Sutton, "Learning to predict by the methods of temporal differences." Machine learning, 3(1):9–44, 1988.](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
8. [M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning." arXiv preprint arXiv:1710.02298, 2017.](https://arxiv.org/pdf/1710.02298.pdf)


