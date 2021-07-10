# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Mixture Density NN models for BayesSim."""

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from ..utils import pdf


class MDNN(nn.Module):
    LL_LIMIT = 1.0e5   # limit log likelihood to avoid large gradients
    MIN_WEIGHT = 1.0e-5  # minimum component weights to enable updates
    EPS_NOISE = 1.e-5  # small noise e.g. for numerical stability

    def __init__(self, input_dim, output_dim, output_lows, output_highs,
                 n_gaussians, hidden_layers, lr, activation, full_covariance,
                 device='cpu', **kwargs):
        """Constructs and initializes a Mixture Density Network.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input
        output_dim : int
            Dimensionality of the output
        n_gaussians : int
            Number of Gaussian components for the mixture
        hidden_layers : list or tuple of int
            Size of each fully-connected hidden layer for the main NN
        output_lows: array
            Flat array of lows for output ranges
        output_highs: array
            Flat array of highs for output ranges
        activation: Module
            torch.nn activation class, e.g. Tanh, LeakyReLU
        lr: float
            Learning rate for the optimizer
        device : string
            Device string (e.g. 'cpu', 'cuda:0')
        """
        super(MDNN, self).__init__()
        self.output_dim = output_dim
        self.output_lows = None
        self.output_highs = None
        if output_lows is not None:
            self.output_lows = torch.from_numpy(output_lows).float().to(device)
            self.output_highs = torch.from_numpy(output_highs).float().to(device)
        self.n_gaussians = n_gaussians
        self.lr = lr
        self.device = device
        self.dropout_p = 0.1
        # Construct the main part of NN as nn.Sequential
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        net = OrderedDict()
        last_layer_size = input_dim
        for l, layer_size in enumerate(hidden_layers):
            net['fcon%d' % l] = nn.Linear(last_layer_size, layer_size)
            net['fdrop%d' % l] = nn.Dropout(self.dropout_p)
            net['nl%d' % l] = activation()
            last_layer_size = layer_size
        self.net = nn.Sequential(net) if len(hidden_layers) > 0 else None
        self.pi = nn.Linear(last_layer_size, n_gaussians)  # mixture weights
        self.mu = nn.Linear(last_layer_size, output_dim * n_gaussians)  # means
        # Components for computing covariance matrices: self.Diag outputs
        # the main diagonal; self.Lower outputs the lower triangular entries.
        self.Diag = nn.Sequential(
            nn.Linear(last_layer_size, output_dim * n_gaussians))
            # nn.Hardtanh(-3, 3))
        self.Lower = None
        self.L_size = int(0.5 * output_dim * (output_dim - 1))
        if self.L_size > 0 and full_covariance:
            self.Lower = nn.Linear(last_layer_size, self.L_size * n_gaussians)
        self.to(self.device)  # move all member variables to device

    def forward(self, x):
        """Applies NNs to the input and outputs weight, mean, variance info.

        Parameters
        ----------
        x : torch.Tensor
            A batch of input vectors

        Returns
        -------
        weights: torch.Tensor
            Mixture weights
        mu: torch.Tensor
            Mixture means
        L_d: torch.Tensor
            Covariance diagonals for each component
        L: torch.Tensor
            Lower triangular factors for each component
        """
        net_out = self.net(x) if self.net is not None else x
        weights = nn.functional.softmax(self.pi(net_out), -1)
        weights = torch.clamp(weights, MDNN.MIN_WEIGHT, 1.0)
        weights /= torch.sum(weights, dim=1, keepdim=True)  # re-normalize
        mu = self.mu(net_out).reshape(-1, self.output_dim, self.n_gaussians)
        L_d = torch.exp(self.Diag(net_out)).reshape(
            -1, self.output_dim, self.n_gaussians)
        eps = MDNN.EPS_NOISE*L_d.mean()
        L_d = L_d + torch.rand_like(L_d).detach()*eps
        L = None
        if self.Lower is not None:
            L = self.Lower(net_out).reshape(-1, self.L_size, self.n_gaussians)
        assert(torch.isfinite(weights).all())
        assert(torch.isfinite(mu).all())
        assert(torch.isfinite(L_d).all())
        if L is not None:
            assert(torch.isfinite(L).all())
        return weights, mu, L_d, L

    def mdn_loss_fn(self, weights, mu, L_d, L, y):
        """Computes loss for training MDN.

        Parameters
        ----------
        weights: torch.Tensor
            Mixture weights
        mu: torch.Tensor
            Mixture means
        L_d: torch.Tensor
            Covariance diagonals for each component
        L: torch.Tensor
            Lower triangular factors for each component
        y: torch.Tensor
            target values

        Returns
        -------
        loss: torch.Tensor
            Loss for training MDN.
        """
        batch_size = y.size()[0]
        result = torch.zeros(batch_size, self.n_gaussians).to(self.device)
        tril_ids = np.tril_indices(self.output_dim, -1)
        for comp_id in range(self.n_gaussians):
            scale_tril = torch.diag_embed(
                L_d[:, :, comp_id], offset=0, dim1=-2, dim2=-1)
            if L is not None:
                scale_tril[:, tril_ids[0], tril_ids[1]] = L[:, :, comp_id]
            gaussian = MultivariateNormal(
                    loc=mu[:, :, comp_id], scale_tril=scale_tril)
            gauss_part = gaussian.log_prob(y)
            gauss_part = torch.clamp(gauss_part, -MDNN.LL_LIMIT, MDNN.LL_LIMIT)
            weight_part = torch.clamp(weights[:, comp_id], MDNN.MIN_WEIGHT, 1.0)
            result[:, comp_id] = gauss_part + weight_part.log()
            if not torch.isfinite(gauss_part).all():
                print('y', torch.isfinite(y).all(), y)
                print('comp_id', comp_id)
                print('weights', weights[:, comp_id],
                      torch.isfinite(weights[:, comp_id].log()).all())
                print('mu', mu[:, :, comp_id],
                      torch.isfinite(mu[:, :, comp_id]).all())
                print('scale_tril', scale_tril,
                      torch.isfinite(scale_tril).all())
                print('gauss_part', gauss_part)
            assert(torch.isfinite(gauss_part).all())
            assert(torch.isfinite(weight_part).all())
            assert(torch.isfinite(result).all())
        loss = -1.0*torch.logsumexp(result, dim=1)
        # Pay attention to the points with the largest loss.
        # loss, _ = torch.topk(loss, k=max(1, int(0.1*batch_size)), largest=True)
        return loss.mean()

    def run_training(self, x_data, y_data, n_updates, batch_size, test_frac=0.2):
        """Runs MDN training.

        Parameters
        ----------
        x_data : torch.Tensor
            Input data
        y_data: torch.Tensor
            Target output data
        n_updates: int
            Number of gradient steps/updates used for neural network training
        batch_size: int
            Batch size for neural network training
        test_frac: float
            Fraction of dataset to keep as test

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged during training
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.output_lows is not None:
            y_data = self.normalize_samples(y_data)
        n_tot = x_data.shape[0]
        n_train = max(int(n_tot*(1.0 - test_frac)), 1)
        x_train_data = x_data[:n_train].to(self.device)
        y_train_data = y_data[:n_train].to(self.device)
        x_test_data = x_data[n_train:].to(self.device)
        y_test_data = y_data[n_train:].to(self.device)

        # We retain all training data in memory to avoid moving out of GPU.
        # PyTorch implementations often use a data loader that keeps large
        # datasets on disk/CPU and then loads them into GPU memory during
        # training. This could be appropriate for simulators that run on CPU,
        # but for IG we would like to avoid this overhead, since IG simulation
        # can be run directly on the GPU.
        def batch_generator():
            while True:
                ids = np.random.randint(0, len(x_train_data), batch_size)
                yield x_train_data[ids], y_train_data[ids]

        batch_gen_iter = batch_generator()

        train_loss_list = []
        test_loss_list = []
        for epoch in range(n_updates):
            x_batch, y_batch = next(batch_gen_iter)
            optimizer.zero_grad()
            pi, mu, L_d, L = self(x_batch)
            loss = self.mdn_loss_fn(pi, mu, L_d, L, y_batch)
            loss.backward()
            optimizer.step()
            if epoch%max(n_updates // 10, 1) == 0 or epoch + 1 == n_updates:
                test_loss = self.mdn_loss_fn(
                    *self(x_test_data), y_test_data)
                test_loss = test_loss.item()
                train_loss_list.append(loss.item())
                test_loss_list.append(test_loss)
                print(f'loss: train {loss.item():0.4f}'
                      f' test {test_loss:0.4f}')
        return {'train_loss': train_loss_list, 'test_loss': test_loss_list}

    def normalize_samples(self, params):
        rng = self.output_highs - self.output_lows
        normed_params = (params - self.output_lows) / rng
        return normed_params

    def predict_MoGs(self, xs):
        """Returns the conditional mixture of Gaussians at each point xs[pt].

        Parameters
        ----------
        xs : torch.Tensor
            A batch of input data

        Returns
        -------
        mogs : list of utils/pdf.MoG
            A list of mixtures (one for each row in xs)
        """
        ntest, dim = xs.size()
        pi, mu, L_d, L = self(xs)
        rng = self.output_highs - self.output_lows
        tril_ids = np.tril_indices(self.output_dim, -1)
        mogs = []
        pi = pi.detach().cpu().numpy()
        normalize = self.output_lows is not None
        for pt in range(ntest):
            Ls = []
            ms = []
            for comp_id in range(self.n_gaussians):
                m = mu[pt, :, comp_id]
                if normalize:
                    m = m*rng + self.output_lows  # denorm
                ms.append(m.detach().cpu().numpy())
                Lwr = torch.diag_embed(
                    L_d[pt, :, comp_id], offset=0, dim1=-2, dim2=-1)
                if L is not None:
                    Lwr[tril_ids[0], tril_ids[1]] = L[:, :, comp_id]
                if normalize:  # denorm to get Rng*Lwr*Lwr^T*Rng^T
                    Lwr = torch.matmul(torch.diag(rng), Lwr)
                L_combo = torch.diag(Lwr)
                if L is not None:
                    L_combo = torch.cat([L_combo, Lwr[tril_ids]], dim=-1)
                Ls.append(L_combo.detach().cpu().numpy())
            mogs.append(pdf.MoG(a=pi[pt, :], ms=ms, Ls=Ls))
        return mogs
