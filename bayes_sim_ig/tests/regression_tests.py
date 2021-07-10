# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Regression tests for BayesSim training.

python -m bayes_sim_ig.tests.regression_tests
"""

import os
from os.path import dirname as dirup

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..bayes_sim import BayesSim
from ..utils import plot
from ..utils.summarizers import *


DATA_FILE_NAMES = [
    {'sim': 'pendulum_train_data_ones_policy_rnd.npz',
     'true': 'pendulum_true_data_ones_policy_rnd.npz'},
    {'sim': 'pendulum_train_data_ones_policy_nornd.npz',
     'true': 'pendulum_true_data_ones_policy_nornd.npz'},]


def load_pendulum_data(fnm, summarizer_fxn):
    loaded = np.load(fnm)
    params = torch.from_numpy(loaded['params']).float()
    states_acts = torch.from_numpy(loaded['data']).float()
    if len(params.shape) == 1:  # reshape if only one data point
        params = params.reshape(1, -1)
        states_acts = states_acts.reshape(1, -1)
    print('Loaded params', params.shape, 'states_acts', states_acts.shape)
    if summarizer_fxn is None:
        return params, states_acts  # raw traj
    state_sz = 3  # cos(theta), sin(theta), thetadot
    states_acts = torch.from_numpy(loaded['data']).float().reshape(
        params.shape[0], -1, state_sz+1)
    print('Reshaped states_acts', states_acts.shape)
    if '_batch' in summarizer_fxn.__name__:
        traj_summaries = summarizer_fxn(states_acts)
    else:
        traj_summaries = []
        for tmp_i in range(states_acts.shape[0]):
            traj_summaries.append(summarizer_fxn(
                states_acts[tmp_i, :, :state_sz],
                states_acts[tmp_i, :, state_sz:]))
        traj_summaries = torch.stack(traj_summaries)
    print('traj_summaries', traj_summaries.shape)
    return params, traj_summaries


def run_tests(model_class, device, data_file_names, summarizer_fxn):
    sim_params_names = np.array(['length', 'mass'])  # need np array format
    n_sim_params = len(sim_params_names)
    sim_params_lows = np.array([0.01]*n_sim_params)
    sim_params_highs = np.array([2.0]*n_sim_params)
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    sim_params, sim_traj_summaries = load_pendulum_data(
        os.path.join(data_dir, data_file_names['sim']), summarizer_fxn)
    if 'nornd' in data_file_names['sim']:
        n_traj = 1250  # n_traj*0.8=1000 training trajs as in original code
        sim_params = sim_params[:n_traj]
        sim_traj_summaries = sim_traj_summaries[:n_traj]
        hidden_layers = (24, 24)
        n_updates = 2000
    else:
        hidden_layers = (128, 128)
        n_updates = 4000
    print('Train BayesSim', model_class, 'with summarizer', summarizer_fxn,
          'on sim_traj_summaries', sim_traj_summaries.shape, 'from data in',
          data_file_names['sim'])
    model_cfg = {'modelClass': model_class, 'components': 10,
                 'hiddenLayers': hidden_layers, 'lr': 1e-3}
    bsim = BayesSim(model_cfg=model_cfg,
                    traj_summaries_dim=sim_traj_summaries.shape[1],
                    params_dim=sim_params.shape[1],
                    params_lows=sim_params_lows, params_highs=sim_params_highs,
                    prior=None, proposal=None, device=device)
    print(bsim.model)
    bsim.run_training(sim_params, sim_traj_summaries,
                      n_updates=n_updates, batch_size=50, test_frac=0.2)
    real_params, real_x = load_pendulum_data(
        os.path.join(data_dir, data_file_names['true']), summarizer_fxn)
    print('BayesSim predict for data with true params', real_params)
    print('real_params\n', real_params, '\nreal_x\n', real_x)
    bsim_params_distr = bsim.predict(real_x)
    print('Posterior bsim_params_distr', bsim_params_distr)
    print('real_params nll', -1.0*bsim_params_distr.eval(
        real_params.detach().cpu().numpy(), log=True, debug=True))
    policy_name = 'nornd' if 'nornd' in data_file_names['sim'] else 'rnd'
    output_file = os.path.join(
        dirup(os.path.realpath(__file__)),
        'BayesSim_regression_test_'+model_class+'_'+
        summarizer_fxn.__name__+'_policy_'+policy_name+'.png')
    print('Making posterior plot in', output_file)
    plot.plot_posterior(
        writer=None, tb_msg=None, tb_step=None,
        sim_params_names=sim_params_names, skip_ids=[],
        true_params=real_params[0], posterior=bsim_params_distr,
        p_lower=sim_params_lows, p_upper=sim_params_highs,
        output_file=output_file)


if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=4,
                        suppress=True, threshold=10000, formatter=None)
    gpu = None
    seed = 2
    np.random.seed(seed)
    use_cuda = (gpu is not None) and torch.cuda.is_available()
    device = 'cuda:'+str(gpu) if use_cuda else 'cpu'
    torch.manual_seed(seed)  # same seed for CUDA to get same model weights
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.deterministic = True   # more reproducible
        torch.cuda.manual_seed_all(seed)
    summarizers = [summary_start, summary_waypts,
                   summary_corr, summary_corrdiff]
    try:
        import signatory
        summarizers.append(summary_signatory_batch)
    finally:
        pass  # signatory is advanced experimental functionality, not required
    for summarizer_fxn in summarizers:
        for data_file_names in DATA_FILE_NAMES:
            for model_class in ['MDNN', 'MDRFF']:
                run_tests(model_class, device, data_file_names, summarizer_fxn)
