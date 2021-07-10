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


class BayesSim(object):
    NUM_TRAIN_TRAJ = 1000

    def __init__(self,
                 model_cfg,
                 traj_summaries_dim,
                 params_dim,
                 params_lows,
                 params_highs,
                 prior,
                 proposal=None,
                 device='cpu'):
        """ Creates and initializes BayesSim object.

        Parameters
        ----------
        model_cfg : model config from yaml file
        traj_summaries_dim : int
            Dimensionality of the input summaries
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

    def run_training(self, params, traj_summaries, n_updates, batch_size,
                     test_frac):
        """Runs the BayesSim algorithm training.

        Parameters
        ----------
        params : torch.Tensor
            Simulation parameters data
        traj_summaries: torch.Tensor
            Trajectory summaries
        n_updates: int
            Number of gradient updates used for neural network training
        batch_size: int
            Batch size for neural network training
        test_frac: float
            Fraction of dataset to keep as test

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged during training
        """
        log_dict = self.model.run_training(
            x_data=traj_summaries, y_data=params, n_updates=n_updates,
            batch_size=batch_size, test_frac=test_frac)
        return log_dict

    def predict(self, xs, threshold=0.005):
        """Predicts posterior given x.

        Parameters
        ----------
        xs : torch.Tensor
            Stats for which to compute the posterior
        threshold: float
            A threshold for pruning negligible mixture components.

        Returns
        -------
        posterior : MoG
            A mixture posterior
        """
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
        # Re-sample MoGs.
        mog_smpls = None
        n_smpls_per_mog = int(1e5/xs.shape[0])
        for tmp_i in range(xs.shape[0]):
            print(mogs[tmp_i])
            new_smpls = mogs[tmp_i].gen(n_samples=n_smpls_per_mog)
            if mog_smpls is None:
                mog_smpls = new_smpls
            else:
                mog_smpls = np.concatenate([mog_smpls, new_smpls], axis=0)
        # Fit a single MoG to compute the final posterior.
        # Remove repeated entries to avoid 'singular' error
        print('Fit MoG from', mog_smpls.shape[0], 'generated samples')
        mog_smpls = np.unique(mog_smpls, axis=0)
        fitted_mog = pdf.fit_mog(mog_smpls, n_components=mogs[0].n_components,
                                 tol=1e-7, maxiter=1000)
        return fitted_mog
