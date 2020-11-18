import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math

from functools import partial

init = False


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    if not isinstance(module, nn.Conv2d):
        return
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_uniform_init(module, gain=1, bias=0):
    if not isinstance(module, nn.Linear):
        return
    nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def DQN(env, args):
    global init
    if args.distributional:
        if args.dueling:
            model = CategoricalDuelingDQN(env, args.noisy, args.device, args.sigma_init,
                                          args.Vmin, args.Vmax, args.num_atoms, args.batch_size)
        else:
            model = CategoricalDQN(env, args.noisy, args.device, args.sigma_init,
                                   args.Vmin, args.Vmax, args.num_atoms, args.batch_size)
    else:
        if args.dueling:
            model = DuelingDQN(env, args.noisy,args.device,  args.sigma_init)
        else:
            model = DQNBase(env, args.noisy,args.device,  args.sigma_init)
    if args.para_init:
        init = True
    return model


class DQNBase(nn.Module):
    """
    Basic DQN + NoisyNet
    Noisy Networks for Exploration
    https://arxiv.org/abs/1706.10295

    parameters
    ---------
    env         environment(openai gym)
    noisy       boolean value for NoisyNet.
                If this is set to True, self.Linear will be NoisyLinear module
    """

    def __init__(self, env, noisy, device, sigma_init):
        super(DQNBase, self).__init__()

        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.noisy = noisy

        if noisy:
            self.Linear = partial(NoisyLinear2, device=device, sigma_init=sigma_init)
        else:
            self.Linear = nn.Linear

        self.flatten = Flatten()

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, self.num_actions)
        )
        global init
        if init:
            self.init_weights()

    def init_weights(self):
        for layer in self.fc:
            xavier_uniform_init(layer)
        for layer in self.features:
            kaiming_init(layer)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon or self.noisy:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action

    def update_noisy_modules(self):
        if self.noisy:
            self.noisy_modules = [module for module in self.modules() if isinstance(module, NoisyLinear2)]

    def sample_noise(self):
        for module in self.noisy_modules:
            module.sample_noise()

    def remove_noise(self):
        for module in self.noisy_modules:
            module.remove_noise()


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """

    def __init__(self, env, noisy, device, sigma_init):
        super(DuelingDQN, self).__init__(env, noisy, device, sigma_init)

        global init
        self.advantage = self.fc

        self.values = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, 1)
        )
        if init:
            # print("init", self.values)
            for layer in self.values:
                xavier_uniform_init(layer)

    # def init_weights(self):
    #     print(self.values)
    #     for layer in self.values:
    #         xavier_uniform_init(layer)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.values(x)
        return value + advantage - advantage.mean(1, keepdim=True)


class CategoricalDQN(DQNBase):
    """
    A Distributional Perspective on Reinforcement Learning
    https://arxiv.org/abs/1707.06887
    """

    def __init__(self, env, noisy, device, sigma_init, Vmin, Vmax, num_atoms, batch_size):
        super(CategoricalDQN, self).__init__(env, noisy, device, sigma_init)

        global init
        support = torch.linspace(Vmin, Vmax, num_atoms)
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, num_atoms)

        self.register_buffer('support', support)
        self.register_buffer('offset', offset)
        self.num_atoms = num_atoms

        self.fc = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, self.num_actions * self.num_atoms),
        )

        self.softmax = nn.Softmax(dim=1)
        if init:
            for layer in self.fc:
                xavier_uniform_init(layer)


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, self.num_actions, self.num_atoms)
        return x

    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon or self.noisy:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_dist = self.forward(state)
                q_value = (q_dist * self.support).sum(2)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action


class CategoricalDuelingDQN(CategoricalDQN):

    def __init__(self, env, noisy, device, sigma_init, Vmin, Vmax, num_atoms, batch_size):
        super(CategoricalDuelingDQN, self).__init__(env, noisy, device, sigma_init, Vmin, Vmax, num_atoms, batch_size)

        global init
        self.advantage = self.fc

        self.value = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, num_atoms)
        )
        if init:
            for layer in self.value:
                xavier_uniform_init(layer)

    # def init_weights(self):
    #     for layer in self.value:
    #         xavier_uniform_init(layer)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)

        advantage = self.advantage(x).view(-1, self.num_actions, self.num_atoms)
        value = self.value(x).view(-1, 1, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, self.num_actions, self.num_atoms)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.register_buffer('sample_weight_in', torch.FloatTensor(in_features))
        self.register_buffer('sample_weight_out', torch.FloatTensor(out_features))
        self.register_buffer('sample_bias_out', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.sample_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.bias_sigma.size(0)))

    def sample_noise(self):
        self.sample_weight_in = self._scale_noise(self.sample_weight_in)
        self.sample_weight_out = self._scale_noise(self.sample_weight_out)
        self.sample_bias_out = self._scale_noise(self.sample_bias_out)

        self.weight_epsilon.copy_(self.sample_weight_out.ger(self.sample_weight_in))
        self.bias_epsilon.copy_(self.sample_bias_out)

    def _scale_noise(self, x):
        x = x.normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyLinear2(nn.Module):
    """Factorised Gaussian NoisyNet"""
    def __init__(self, in_features, out_features, device, sigma_init=0.4):
        super(NoisyLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.noise_std = sigma_init / math.sqrt(self.in_features)
        self.in_noise = torch.FloatTensor(in_features).to(device)
        self.out_noise = torch.FloatTensor(out_features).to(device)
        self.noise = None
        self.sample_noise()

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        normal_y =  nn.functional.linear(x, self.weight, self.bias)
        if not self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * Variable(self.noise)
        noisy_bias = self.noisy_bias * Variable(self.out_noise)
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


