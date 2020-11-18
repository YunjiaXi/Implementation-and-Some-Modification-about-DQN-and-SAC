import  os


def tuned_pong(args):
    args.env_name = 'PongNoFrameskip-v4'
    args.batch_size = 32
    args.buffer_size = 100000
    args.max_frames = 1000000
    args.update_target = 1000
    args.train_freq = 1
    args.learning_start = 10000
    args.eps_final = 0.02
    args.eps_decay = 100000
    args.normal_init = False
    args.max_eps_step = 27000
    args.logdir = 'runs_pong'


def tuned_boxing(args):
    args.env_name = 'BoxingNoFrameskip-v4'
    args.batch_size = 32
    args.buffer_size = 3000000
    args.update_target = 1000
    args.max_frames = 30000000
    args.train_freq = 4
    args.learning_start = 10000
    args.eps_final = 0.01
    args.eps_decay = 1000000
    args.max_eps_step = 27000
    args.seed = 101
    args.normal_init = True
    args.logdir = 'runs_boxing'


def tuned_breakout(args):
    args.env_name = 'BreakoutNoFrameskip-v4'
    args.batch_size = 32
    args.buffer_size = 10000000
    args.update_target = 10000
    args.max_frames = 50000000
    args.train_freq = 4
    args.learning_start = 10000
    args.eps_final = 0.01
    args.eps_decay = 1000000
    args.max_eps_step = 4500
    args.seed = 100
    args.normal_init = True
    args.logdir = 'runs_breakout'
