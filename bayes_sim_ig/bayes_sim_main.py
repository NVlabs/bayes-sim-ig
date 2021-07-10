# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""A demo of using BayesSim.""

Example commands:
export ISAACGYM_PATH /path/to/IsaacGym/code && \
python -m bayes_sim_ig.bayes_sim_main --task Pendulum --headless \
  --logdir /tmp/tmp_learn/ --max_iterations 20 --sim_device cpu --rl_device cpu

python -m bayes_sim_ig.bayes_sim_main --task Ant --headless \
  --logdir /tmp/tmp_learn/ --max_iterations 20 \
  --sim_device cuda:0 --rl_device cuda:0

View Tensorboard results with:
tensorboard --logdir=/tmp/tmp_learn/ --bind_all --port 6008 \
  --samples_per_plugin images=1000

If RL is run from scratch after every BayesSim iteration (i.e. ftuneRL==False
in the config file), then Tensorboard would show many subdirectories with logs
for each RL run separately.
To view only the BayesSim visualizations type 'bsim' under Runs after launching
Tensorboard (to replace gray text that says 'Write regex to filter runs').
To select a subset of RL logs use, for example:
^Ant.*/rl_[0,3]$
This will show RL logs for runs 0 and 3 for the Ant runs.
"""
import gc
import os
import sys

from rlgpu.utils.config import set_seed
from rlgpu.utils.process_ppo import process_ppo

import numpy as np
np.set_printoptions(edgeitems=30,  linewidth=4000,  precision=4,
                    infstr='inf', nanstr='nan', suppress=True,
                    threshold=10000, formatter=None)
import torch  # torch and anything that uses it has to be imported after IG
from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(edgeitems=30, linewidth=1000, threshold=10000,
                       sci_mode=False)

from .bayes_sim import BayesSim
from .sim.ig_env_wrappers import make_ig_env
from .utils import pdf, plot
from .utils.args import init_args, log_args, check_distr, load_real_params
from .utils.collect_trajectories import *  # used dynamically
from .utils.summarizers import *  # used dynamically


def main():
    #
    # Init args. Create IG a vectorized IG env/task.
    #
    args, cfg_env, cfg_train = init_args()
    set_seed(cfg_train['seed'])
    env = make_ig_env(args, cfg_env, cfg_train)
    params_gen = env.task.actor_params_generator
    #
    # Init real and sim parameter distributions.
    #
    eal_weights, real_means, real_stds = load_real_params(cfg_env, params_gen)
    real_params_distr = pdf.MoG(a=eal_weights, ms=real_means, Ss=real_stds)
    check_distr(real_params_distr, params_gen.lows, params_gen.highs,
                'realParams')
    print('Init real_params_distr', real_params_distr)
    sim_params_distr = pdf.Uniform(params_gen.lows, params_gen.highs)  # prior
    params_gen.set_distr(sim_params_distr)
    print('Init sim_params_distr', sim_params_distr)
    writer = SummaryWriter(
        log_dir=os.path.join(args.logdir, 'bsim'), flush_secs=10)
    log_args(args, cfg_env, cfg_train, writer)
    #
    # Init PPO.
    #
    ftune_rl = cfg_env['bayessim']['ftuneRL']
    ppo = process_ppo(
        args, env, cfg_train,
        os.path.join(args.logdir, 'rl_0') if not ftune_rl else args.logdir)
    #
    # Main 'real' loop.
    #
    summarizer_fxn = eval(cfg_env['bayessim']['summarizerFxn'])
    n_train_trajs = cfg_env['bayessim']['trainTrajs']
    if 'policyCheckpt' in cfg_env['bayessim']:
        ppo.load(cfg_env['bayessim']['policyCheckpt'])
    collect_policy_fxn = eval(cfg_env['bayessim']['collectPolicy'])
    bsim = None
    bsim_model_class = cfg_env['bayessim']['modelClass']
    bsim_ftune = cfg_env['bayessim']['finetuneUpdates'] > 0
    for real_iter_id in range(cfg_env['bayessim']['realIters']):
        #
        # Plot sim_params_distr that we are about to use for PPO training
        #
        figs = plot.plot_posterior(
            writer, 'BayesSim/posterior', real_iter_id,
            sim_params_names=params_gen.names,
            skip_ids=params_gen.skip_ids,
            true_params=real_params_distr.components[0].m,
            posterior=sim_params_distr,
            p_lower=params_gen.lows, p_upper=params_gen.highs)
        #
        # Train PPO on the current simulation parameter posterior.
        #
        print('============= Train RL before real_iter_id', real_iter_id)
        params_gen.set_distr(sim_params_distr)
        if not ftune_rl and real_iter_id > 0:  # re-init PPO if needed
            ppo_logdir = os.path.join(args.logdir, 'rl_'+str(real_iter_id))
            ppo = process_ppo(args, env, cfg_train, ppo_logdir)
            print('RL restart with logdir', ppo_logdir)
            ppo.run(num_learning_iterations=args.max_iterations,
                    log_interval=cfg_train['learn']['save_interval'])
        else:
            ppo.vec_env.reset()  # overloaded method that does reset all envs
            ppo_it = real_iter_id*args.max_iterations
            ppo.current_learning_iteration = ppo_it
            ppo.run(num_learning_iterations=ppo_it+args.max_iterations,
                    log_interval=cfg_train['learn']['save_interval'])
        #
        # Simulate running on a real robot with real parameters.
        #
        print('Simulating evals...')
        params_gen.set_distr(real_params_distr)
        num_eval_episodes = cfg_env['bayessim']['realEvals']
        _, _, real_rwds, real_imgs = collect_trajectories(
            num_eval_episodes, ppo, None,
            None, max_traj_len=None, device='cpu',
            verbose=False, visualize=True)
        for fxn in ['mean', 'min', 'max']:
            writer.add_scalar('SurrogateReal/real_rewards_'+fxn,
                              eval('torch.'+fxn)(real_rwds), real_iter_id)
        if len(real_imgs) > 0:
            vid_imgs = torch.tensor(real_imgs)
            vid_imgs = vid_imgs.unsqueeze(0).transpose(4, 3).transpose(3, 2)
            writer.add_video('RealSurrogate/video', vid_imgs, real_iter_id, 24)
        if bsim_model_class == 'None':
            continue  # ablation: no BayesSim
        #
        # Collect trajectories for BayesSim training and train BayesSim.
        #
        print('Collect trajectories for BayesSim', bsim_model_class)
        curr_n_train_trajs = n_train_trajs
        n_updates = cfg_env['bayessim']['trainUpdates']
        if bsim_ftune and real_iter_id > 0:
            ftune_n_updates = cfg_env['bayessim']['finetuneUpdates']
            curr_n_train_trajs *= (float(ftune_n_updates)/n_updates)
            n_updates = ftune_n_updates
        unif = pdf.Uniform(params_gen.lows, params_gen.highs)
        params_gen.set_distr(unif)
        sim_params_smpls, sim_traj_summaries, _, _ = collect_trajectories(
            n_train_trajs, ppo, summarizer_fxn, collect_policy_fxn,
            max_traj_len=cfg_env['bayessim']['trainTrajLen'],
            device=args.rl_device, verbose=True, visualize=False)
        print('Train BayesSim', bsim_model_class)
        if bsim is None or not bsim_ftune:
            bsim = BayesSim(
                model_cfg=cfg_env['bayessim'],
                traj_summaries_dim=sim_traj_summaries.shape[1],
                params_dim=sim_params_smpls.shape[1],
                params_lows=params_gen.lows,
                params_highs=params_gen.highs,
                prior=None, proposal=None, device=args.rl_device)
        log_bsim = bsim.run_training(
            sim_params_smpls, sim_traj_summaries, n_updates=n_updates,
            batch_size=cfg_env['bayessim']['trainBatchSize'], test_frac=0.2)
        del sim_params_smpls
        del sim_traj_summaries
        gc.collect()
        torch.cuda.empty_cache()
        #
        # Update posterior using surrogate real trajs and BayesSim inference.
        #
        print('Simulating surrogate real runs...')
        params_gen.set_distr(real_params_distr)
        real_params, real_xs, real_rwds, _ = collect_trajectories(
            cfg_env['bayessim']['realTrajs'], ppo,
            summarizer_fxn, collect_policy_fxn,
            max_traj_len=cfg_env['bayessim']['trainTrajLen'],  # match train
            device=args.rl_device, verbose=False, visualize=False)
        sim_params_distr = bsim.predict(real_xs.to(args.rl_device))
        writer.add_scalar(
            'BayesSim/train_loss', log_bsim['train_loss'][-1], real_iter_id)
        writer.add_scalar(
            'BayesSim/test_loss', log_bsim['test_loss'][-1], real_iter_id)
        writer.flush()
        sys.stdout.flush()


if __name__ == '__main__':
    main()
