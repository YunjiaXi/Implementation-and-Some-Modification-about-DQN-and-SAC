import math, os
import datetime, time
import pathlib
import random
import torch
import numpy as np


def find_gpu():
    import numpy as np
    pid = os.getpid()
    tmp_file = 'tmp{}'.format(pid)
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{}'.format(tmp_file))
    memory_gpu = [int(x.split()[2]) for x in open(tmp_file, 'r').readlines()]
    os.system('rm {}'.format(tmp_file))
    return np.argmax(memory_gpu)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret


def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)

    return function


def beta_scheduler(beta_start, beta_frames):
    def function(frame_idx):
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

    return function


def create_log_dir(args):
    log_dir = args.env_name  # + '-eps_decay-' + str(args.eps_decay) + '-eps_final-' + str(args.eps_final)
    if args.multi_step != 1:
        log_dir = log_dir + "{}-step-".format(args.multi_step)
    if args.distributional:
        log_dir = log_dir + "c51-"
    if args.prioritized_replay:
        log_dir = log_dir + "per-"
    if args.dueling:
        log_dir = log_dir + "dueling-"
    if args.double:
        log_dir = log_dir + "double-"
    if args.noisy:
        log_dir = log_dir + "noisy-"
    log_dir = log_dir + "dqn-seed-" + str(args.seed)
    if args.soft_update:
        log_dir += ("-soft_update-tau-" + str(args.tau))
    log_dir += ('-init-para-'+str(args.para_init))

    log_dir += ('-train_freq-' + str(args.train_freq))
    log_dir += ('-buffer-size-' + str(args.buffer_size))

    now = datetime.datetime.now().strftime('-%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + now

    log_dir = os.path.join(args.logdir, log_dir)
    return log_dir


def print_log(frame, prev_frame, prev_time, reward_list, length_list, loss_list):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    avg_reward = np.mean(reward_list)
    avg_length = np.mean(length_list)
    avg_loss = np.mean(loss_list) if len(loss_list) != 0 else 0.

    print("Train -- Frame: {:<8} FPS: {:.2f} Avg. Length: {:.2f}  Avg. Reward: {:.2f} Avg. Loss: {:.2f}".format(
        frame, fps, avg_length, avg_reward, avg_loss
    ))


def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))


def save_model(model, args):
    fname = ""
    if args.multi_step != 1:
        fname += "{}-step-".format(args.multi_step)
    if args.distributional:
        fname += "c51-"
    if args.prioritized_replay:
        fname += "per-"
    if args.dueling:
        fname += "dueling-"
    if args.double:
        fname += "double-"
    if args.noisy:
        fname += "noisy-"
    fname += "dqn-{}.pth".format(args.save_model)
    fname = os.path.join("models", fname)

    pathlib.Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), fname)


def load_model(model, args):
    if args.load_model is not None:
        fname = os.path.join("models", args.load_model)
    else:
        fname = ""
        if args.multi_step != 1:
            fname += "{}-step-".format(args.multi_step)
        if args.distributional:
            fname += "c51-"
        if args.prioritized_replay:
            fname += "per-"
        if args.dueling:
            fname += "dueling-"
        if args.double:
            fname += "double-"
        if args.noisy:
            fname += "noisy-"
        fname += "dqn-{}.pth".format(args.save_model)
        fname = os.path.join("models", fname)

    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None

    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    model.load_state_dict(torch.load(fname, map_location))

