import math
import torch
import datetime, os


def create_log_dir(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = args.logdir + "/" + now + '-' + args.env_name + '-' + args.policy
    if args.automatic_entropy_tuning:
        log_dir += "-autotune"
    else:
        log_dir += ('-alpha' + str(args.alpha))
    log_dir += ('-target_update-' + str(args.target_update_interval))
    log_dir += ('-seed-' + str(args.seed))
    if args.noisy:
        log_dir += '-noisy'
    return log_dir


def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
