import torch
import argparse
import time, os
from torch.utils.tensorboard import SummaryWriter

from utils import create_log_dir, print_args, find_gpu
from tuned_hyperparameter import tuned_boxing, tuned_pong
from wrapper import make_atari, wrap_atari_dqn
from train import train, test


def get_args():
    parser = argparse.ArgumentParser(description='DQN and its variants')

    # Basic Arguments
    parser.add_argument('--seed', type=int, default=1112,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # tuned hyper-parameter for some environment
    parser.add_argument('--tuned-pong', action='store_true', default=False,
                        help='Use tuned hyper-parameters for Pong')
    parser.add_argument('--tuned-boxing', action='store_true', default=False,
                        help='Use tuned hyper-parameters for Boxing')
    # parser.add_argument('--tuned-breakout', action='store_true', default=False,
    #                     help='Use tuned hyper-parameters for Breakout')

    # Training Arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max-eps-step', type=int, default=27000, metavar='STEPS',
                        help='Max steps of an episode')
    parser.add_argument('--max-frames', type=int, default=1500000, metavar='STEPS',
                        help='Number of frames to train')
    parser.add_argument('--buffer-size', type=int, default=100000, metavar='CAPACITY',
                        help='Maximum memory buffer size')
    parser.add_argument('--update-target', type=int, default=1000, metavar='STEPS',
                        help='Interval of target network update')
    parser.add_argument('--train-freq', type=int, default=1, metavar='STEPS',
                        help='Number of steps between optimization step')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--normal-init', type=bool, default=False,
                        help='Whether convolutional layers are initialized with normal')
    parser.add_argument('--learning-start', type=int, default=10000, metavar='N',
                        help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--eps-start', type=float, default=1.0,
                        help='Start value of epsilon')
    parser.add_argument('--eps-final', type=float, default=2e-2,
                        help='Final value of epsilon')
    parser.add_argument('--eps-decay', type=int, default=100000,
                        help='Adjustment parameter for epsilon')
    parser.add_argument('--soft-update', type=bool, default=False,
                        help='Whether use soft update(default: False)')
    parser.add_argument('--tau', type=int, default=0.005,
                        help='target smoothing coefficient(τ) (default: 0.005)')

    # Algorithm Arguments
    parser.add_argument('--double', action='store_true',
                        help='Enable Double-Q Learning')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--noisy', action='store_true',
                        help='Enable Noisy Network')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='Enable prioritized experience replay')
    parser.add_argument('--distributional', action='store_true',
                        help='Enable categorical dqn')
    parser.add_argument('--multi-step', type=int, default=1,
                        help='N-Step Learning')
    parser.add_argument('--Vmin', type=int, default=-10,
                        help='Minimum value of support for c51')
    parser.add_argument('--Vmax', type=int, default=10,
                        help='Maximum value of support for c51')
    parser.add_argument('--num-atoms', type=int, default=51,
                        help='Number of atom for c51')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha value for prioritized replay')
    parser.add_argument('--beta-start', type=float, default=0.4,
                        help='Start value of beta for prioritized replay')
    parser.add_argument('--beta-frames', type=int, default=100000,
                        help='End frame of beta schedule for prioritized replay')
    parser.add_argument('--sigma-init', type=float, default=0.4,
                        help='Sigma initialization value for NoisyNet')
    parser.add_argument('--para-init', type=bool, default=False,
                        help='Sigma initialization value for NoisyNet')

    # Environment Arguments
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4',
                        help='Environment Name')
    parser.add_argument('--episode-life', type=int, default=1,
                        help='Whether env has episode life(1) or not(0)')
    parser.add_argument('--clip-rewards', type=int, default=1,
                        help='Whether env clip rewards(1) or not(0)')
    parser.add_argument('--frame-stack', type=int, default=1,
                        help='Whether env stacks frame(1) or not(0)')
    parser.add_argument('--scale', type=int, default=0,
                        help='Whether env scales(1) or not(0)')

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, default=None,
                        help='Pretrained model name to load (state dict)')
    parser.add_argument('--save-model', type=str, default='model',
                        help='Pretrained model name to save (state dict)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--eval-time', type=int, default=10,
                        help='evaluate time')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation agent')
    parser.add_argument('--evaluation-interval', type=int, default=10000,
                        help='Frames for evaluation interval')

    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=1e-4, metavar='η',
                        help='Learning rate')

    parser.add_argument('--logdir', default='runs/',
                        help='the folder that store log info')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(find_gpu()) if args.cuda else "cpu")
    return args


def main():
    args = get_args()
    if args.tuned_pong:
        tuned_pong(args)
    elif args.tuned_boxing:
        tuned_boxing(args)
    print_args(args)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)

    env = make_atari(args)
    env = wrap_atari_dqn(env, args)

    # set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        test(env, args)
        env.close()
        return

    train(env, args, writer)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()