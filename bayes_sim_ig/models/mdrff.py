# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Mixture Density NN model with RFF features for BayesSim."""

from .mdnn import MDNN
from .rff import RFF


class MDRFF(MDNN):
    def __init__(self, input_dim, output_dim, output_lows, output_highs,
                 n_gaussians, lr, activation, full_covariance, device='cpu',
                 n_feat=500, kernel='RBF', sigma=1.0, **kwargs):
        super().__init__(n_feat, output_dim, output_lows, output_highs,
                         n_gaussians, hidden_layers=[], lr=lr,
                         activation=activation, full_covariance=full_covariance,
                         device=device)
        self.rff = RFF(n_feat, input_dim, sigma, cos_only=False,
                       quasi_random=False if input_dim > 100 else True,
                       kernel=kernel, device=device)
        print('MDRFF n_feat', n_feat, 'sigma', sigma)
        print(self)

    def forward(self, x):
        x_feat = self.rff.to_features(x)
        return super().forward(x_feat)
