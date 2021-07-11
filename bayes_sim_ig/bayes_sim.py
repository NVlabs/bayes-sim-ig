# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""The core BayesSim class."""

import numpy as np
import torch

from .models.mdnn import MDNN    # used dynamically
from .models.mdrff import MDRFF  # used dynamically
from .utils import pdf
from .utils.summarizers import *  # used dynamically


class BayesSim(object):
    NUM_TRAIN_TRAJ_PER_BATCH = 1000  # num trajs for each training batch
    NUM_TRAIN_EPOCHS = 10            # num times to go over the batch
    MINIBATCH_SIZE = 100             # minibatch size for NN training
    NUM_GRAD_UPDATES = NUM_TRAIN_EPOCHS*NUM_TRAIN_TRAJ_PER_BATCH//MINIBATCH_SIZE

    TEST_FRACTION = 0.2              # fraction of dataset to use as test

    def __init__(self,
                 model_cfg, obs_dim, act_dim,
                 params_dim, params_lows, params_highs,
                 prior, proposal=None, device='cpu'):
        """ Creates and initializes BayesSim object.

        Parameters
        ----------
        model_cfg : bayessim section of the yaml config
        obs_dim: int
            Dimensionality of the environment/task observations/states
        act_dim: int
            Dimensionality of the environment/task actions/controls
        params_dim : int
            Number of simulation parameters to estimate
        params_lows : array
            A flat array with lows for simulation params
        params_highs : array
            A flat array with highs for simulation params
        prior : None or torch.distributions.MixtureSameFamily
            A prior distribution
        proposal : torch.distributions.MixtureSameFamily, optional
            A proposal distribution
        device : string
            Device string (e.g. 'cuda:0')
        """
        self.prior = prior
        self.proposal = proposal
        model_class = model_cfg['modelClass']
        self.summarizer_fxn = eval(model_cfg['summarizerFxn'])
        tmp_smry = self.summarizer_fxn(
            torch.zeros(1, model_cfg['trainTrajLen'], obs_dim),
            torch.zeros(1, model_cfg['trainTrajLen'], act_dim))
        traj_summaries_dim = tmp_smry.shape[-1]
        full_covariance = False
        if 'fullCovariance' in model_cfg:
            full_covariance = model_cfg['fullCovariance'],
        kwargs = {'input_dim': traj_summaries_dim, 'output_dim': params_dim,
                  'output_lows': params_lows, 'output_highs': params_highs,
                  'n_gaussians': model_cfg['components'],
                  'hidden_layers': model_cfg['hiddenLayers'],
                  'lr': model_cfg['lr'],
                  'activation': torch.nn.Tanh,
                  'full_covariance': full_covariance,
                  'device': device}
        if model_class.startswith('MDRFF'):
            kernel = 'RBF'
            sigma = 4.0
            if '_' in model_class:
                model_params = model_class.split('_')
                model_class = model_params[0]
                kernel = model_params[1]
                if len(model_params) > 2:
                    sigma = float(model_params[2])
            kwargs.update({'n_feat': 200, 'sigma': sigma, 'kernel': kernel})
        self.model = eval(model_class)(**kwargs)

    @staticmethod
    def get_n_trajs_per_batch(n_train_trajs, n_train_trajs_done):
        n_trajs_per_batch = BayesSim.NUM_TRAIN_TRAJ_PER_BATCH
        if n_train_trajs_done + n_trajs_per_batch > n_train_trajs:
            n_trajs_per_batch = n_train_trajs - n_train_trajs_done
        return n_trajs_per_batch

    def run_training(self, params, traj_states, traj_actions):
        """Runs the BayesSim algorithm training.

        Parameters
        ----------
        params : torch.Tensor
            Simulation parameters data
        traj_states: torch.Tensor
            Trajectory states
        traj_actions: torch.Tensor
            Trajectory actions

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged during training
        """
        traj_summaries = self.summarizer_fxn(traj_states, traj_actions)
        log_dict = self.model.run_training(
            x_data=traj_summaries, y_data=params,
            n_updates=BayesSim.NUM_GRAD_UPDATES,
            batch_size=BayesSim.MINIBATCH_SIZE,
            test_frac=BayesSim.TEST_FRACTION)
        return log_dict

    def predict(self, states, actions, threshold=0.005):
        """Predicts posterior given x.

        Parameters
        ----------
        states: torch.Tensor
            Trajectory states
        actions: torch.Tensor
            Trajectory actions
        threshold: float (optional)
            A threshold for pruning negligible mixture components.

        Returns
        -------
        posterior : MoG
            A mixture posterior
        """
        xs = self.summarizer_fxn(states, actions)
        mogs = self.model.predict_MoGs(xs)
        if self.proposal is not None:
            # Compute posterior given prior by analytical division step.
            for tmp_i, mog in enumerate(mogs):
                mog.prune_negligible_components(threshold=threshold)
                if isinstance(self.prior, pdf.Uniform):
                    post = mog / self.proposal
                elif isinstance(self.prior, pdf.Gaussian):
                    post = (mog * self.prior) / self.proposal
                else:
                    raise NotImplemented
                mogs[tmp_i] = post
        if len(mogs) == 1:
            return mogs[0]
        # Make a small net to fit the combined mixture.
        kwargs = {'input_dim': 1,  # unconditional MDNN
                  'output_dim': self.model.output_dim,
                  'output_lows': self.model.output_lows.detach().cpu().numpy(),
                  'output_highs': self.model.output_highs.detach().cpu().numpy(),
                  'n_gaussians': self.model.n_gaussians,
                  'hidden_layers': (128, 128),
                  'lr': self.model.lr,
                  'activation': self.model.activation,
                  'full_covariance': self.model.L_size > 0,
                  'device': self.model.device}
        mog_model = MDNN(**kwargs)
        # Re-sample MoGs.
        mog_smpls = None
        tot_smpls = int(1e4)
        n_smpls_per_mog = int(tot_smpls/xs.shape[0])
        for tmp_i in range(xs.shape[0]):
            new_smpls = mogs[tmp_i].gen(n_samples=n_smpls_per_mog)
            if mog_smpls is None:
                mog_smpls = new_smpls
            else:
                mog_smpls = np.concatenate([mog_smpls, new_smpls], axis=0)
        mog_smpls = torch.from_numpy(mog_smpls).float().to(self.model.device)
        # Fit a single MoG to compute the final posterior.
        print(f'Fitting posterior from {len(mogs):d} mogs')
        batch_size = 100
        n_updates = 5*tot_smpls//batch_size
        input = torch.zeros(mog_smpls.shape[0], 1).to(self.model.device)
        mog_model.run_training(input, mog_smpls, n_updates, batch_size)
        fitted_mogs = mog_model.predict_MoGs(input[0:1, :])
        assert(len(fitted_mogs) == 1)
        return fitted_mogs[0]
