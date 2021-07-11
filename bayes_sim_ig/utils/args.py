# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utils for initializing arguments to be appropriate for BayesSim runs."""

from datetime import datetime
import re
import os
from os.path import dirname as dirup
import sys

import numpy as np

from rlgpu.utils.config import get_args, load_cfg

IG_TASKS = ['Ant', 'Anymal', 'BallBalance', 'Cartpole', 'FrankaCabinet',
            'Humanoid', 'Ingenuity', 'Pendulum', 'Quadcoper', 'ShadowHand']

def init_args():
    """Creates args as appropriate for BayesSim runs"""
    # Initial sanity check to ensure that custom yaml configs were provided.
    assert('--task' in sys.argv[1:])
    try:
        args = get_args()  # note: this only works with default tasks
    except:
        idx = sys.argv.index('--task') + 1
        assert(idx < len(sys.argv))
        saved_task = sys.argv[idx]
        sys.argv[idx] = 'Cartpole'  # fake default task to make get_args parse
        args = get_args()
        args.task = saved_task
        sys.argv[idx] = saved_task
    if args.task not in IG_TASKS:
        print('Need one of the supported tasks:', IG_TASKS, 'got', args.task)
        exit(1)
    # Try to load default configs if needed.
    bsim_root_dir = dirup(dirup(os.path.realpath(__file__)))
    pfx = re.findall('[A-Z][^A-Z]*', args.task)
    pfx = '_'.join(pfx).lower()
    if '--cfg_env' not in sys.argv:
        args.cfg_env = os.path.join(bsim_root_dir, 'cfg', pfx+'.yaml')
    if '--cfg_train' not in sys.argv:
        assert(os.getenv('ISAACGYM_PATH') is not None), \
            'Please set Isaac Gym path: export ISAACGYM_PATH=/path/to/isaacgym'
        train_cfg_path = os.path.join(
            os.path.expanduser(os.environ.get('ISAACGYM_PATH')),
            'python', 'rlgpu', 'cfg', 'train', 'rlpt')
        train_cfg_file = os.path.join(
            train_cfg_path, 'pytorch_ppo_'+pfx+'.yaml')
        if not os.path.exists(train_cfg_file):
            train_cfg_file = os.path.join(
                train_cfg_path, 'pytorch_ppo_cartpole.yaml')
        args.cfg_train = train_cfg_file
        print('cfg_train', train_cfg_file)
    args.task_type = 'Python'
    cfg_env, cfg_train, _ = load_cfg(args)  # get configs, ignore logdir
    assert('bayessim' in cfg_env), f'Need BayesSim section in {args.cfg_env:s}'
    assert(cfg_env['task']['randomize']), \
        f'Need task.randomize==True in {args.cfg_env:s}'
    if args.seed is None:
        args.seed = 0  # make sure we have a numeric default seed
    args.logdir = make_logdir_str(args.logdir, args.task, args.seed,
                                  args.max_iterations, cfg_env)
    return args, cfg_env, cfg_train


def make_logdir_str(pfx, task_name, seed, rl_max_iter, cfg):
    """Constructs an informative name for the directory with logs."""
    rest_str = '_'.join(
        [task_name, cfg['bayessim']['modelClass'],
         'ftune' if cfg['bayessim']['ftune'] else 'noftune',
         cfg['bayessim']['summarizerFxn'],
         cfg['bayessim']['collectPolicy'],
         'rl'+str(rl_max_iter),
         'nreal'+str(cfg['bayessim']['realTrajs']),
         'seed'+str(seed),
          # datetime.strftime(datetime.today(), '%y%m%d_%H%M%S')
         ])
    return os.path.join(pfx, rest_str)


def log_args(args, cfg_env, cfg_train, tb_writer):
    """Logs arguments to tensorboard.
    Tensorboard uses markdown-like formatting, hence '  \n' as separator."""
    cfg_dict = {'cfg_env': cfg_env, 'cfg_train': cfg_train}
    all_str = ''
    for k,v in cfg_dict.items():
        all_str += '  \n  \n{:s}='.format(str(k))
        for k2, v2 in v.items():
            if isinstance(v2, dict):
                all_str += '  \n  \n..{:s}='.format(str(k2))
                for k3, v3 in v2.items():
                    all_str += '  \n....{:s}={:s}'.format(str(k3), str(v3))
            else:
                all_str += '  \n  \n..{:s}={:s}'.format(str(k2), str(v2))
    all_str += '  \n  \nargs='
    for member in vars(args):
        all_str += '  \n...{:s}={:s}'.format(
            str(member), str(getattr(args, member)))
    print(all_str)
    tb_writer.add_text('BayesSim/cfg', all_str)

    return args


def load_real_params(cfg_env, params_gen):
    assert('realParams' in cfg_env['env'])
    real_weights = cfg_env['env']['realParams']['weights']
    real_means = [np.array(x) for x in cfg_env['env']['realParams']['means']]
    real_stds = [np.diag(x) for x in cfg_env['env']['realParams']['stds']]
    real_dims = params_gen.lows.shape[0]
    for i in range(len(real_means)):
        if real_means[i].shape[0] == 1:
            real_means[i] = np.tile(real_means[i], real_dims)
    for i in range(len(real_stds)):
        if real_stds[i].shape[0] == 1:
            real_stds[i] = np.diag(np.tile(real_stds[i][0,0], real_dims))
    return real_weights, real_means, real_stds


def check_distr(distr, lows, highs, msg):
    if distr.components[0].m.shape[0] != lows.shape[0]:
        print(msg, 'dim in yaml should be', lows.shape,
              'got', distr.components[0].m.shape)
        assert(False)
    for comp in distr.components:
        if (comp.m < lows).any() or (comp.m > highs).any():
            print(msg, 'invalid mean')
            for i in range(comp.m.shape[0]):
                if comp.m[i] < lows[i] or comp.m[i] > highs[i]:
                    print('dim', i, 'mean', comp.m[i],
                          'low', lows[i], 'high', highs[i])
            assert(False)
