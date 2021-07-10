# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Random Fourier Features (RFF) and RFF kernels.

The code below is adapted from:
https://github.com/PhilippeMorere/EMU-Q/blob/master/features/RandomFourierFeatures.py
The license for the above source is pasted below:

MIT License

Copyright (c) 2018

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
from scipy.special import erfinv
import ghalton
import torch


class RFF:
    """
    Random Fourier Features, Vanilla or quasi-random
    Note: make sure input space is normalised (meaning the range is either in
          [0,1], [-1,1] or within ~one order of magnitude).
    """
    def to_features(self, x):
        pass

    def __init__(self, n_feat, d, sigma, cos_only=False, quasi_random=True,
                 kernel='RBF', device='cpu'):
        """
        Parameters
        ----------
        m: n_feat
            Number of features
        d: int
            Input dimension
        sigma: float or vector of length d
            Feature lengthscale
        cos_only: bool
            Using cos-only formulation of RFF (Default=False)
        quasi_random: bool
            Whether to use quasi-random sequence to generate RFF
        kernel: str
            Type of kernel to approximate: RBF, Laplace/Matern12, Matern32,
            Matern52
        """
        self.n_feat = n_feat
        self.sigma = sigma
        self.d = int(d)
        self.freqs = None
        self.offset = None
        self.a = 1.0
        self.device = device
        if isinstance(sigma, list):
            assert(len(sigma) == d)
            self.sigma = np.array(sigma, dtype=np.float32)
        else:
            self.sigma = np.ones(d, dtype=np.float32) * sigma
        self.sigma = torch.from_numpy(self.sigma).float().reshape(1, -1)
        self.sigma = self.sigma.to(self.device)
        # Instantiate the kernel object.
        if kernel == "RBF":
            rff_kernel = RFFKernelRBF()
        elif kernel == "Laplace" or kernel == "Matern12":
            rff_kernel = RFFKernelMatern12()
        elif kernel == "Matern32":
            rff_kernel = RFFKernelMatern32()
        elif kernel == "Matern52":
            rff_kernel = RFFKernelMatern52()
        else:
            raise ValueError("Kernel {} is not recognised.".format(kernel))
        # Define feature extraction function and sample frequencies.
        if cos_only:  # cos only features
            self.freqs = RFF.draw_freqs(rff_kernel, n_feat, d, quasi_random)
            self.offset = torch.from_numpy(
                2.0 * np.pi * np.random.rand(1, n_feat)).float().to(self.device)
            self.a = np.sqrt(1.0/float(n_feat))
            self.to_features = self._to_cos_only_features
        else:  # cos and sin features
            assert(self.n_feat % 2 == 0)
            self.freqs = RFF.draw_freqs(rff_kernel, n_feat//2, d, quasi_random)
            self.a = np.sqrt(1.0/float(n_feat/2))
            self.to_features = self._to_cos_sin_features
        self.freqs = torch.from_numpy(self.freqs).float().to(self.device)

    @staticmethod
    def draw_freqs(rff_kernel, m, d, quasi_random):
        if quasi_random:
            perms = ghalton.EA_PERMS[:d]
            sequencer = ghalton.GeneralizedHalton(perms)
            points = np.array(sequencer.get(m+1))[1:]
            freqs = rff_kernel.inv_cdf(points)
        else:
            freqs = rff_kernel.sample_freqs((m, d))
        return freqs

    def _to_cos_only_features(self, x, sigma=None):
        sigma = (self.sigma if sigma is None else sigma)
        coeff = self.freqs / sigma
        inner = torch.matmul(x, coeff.T)
        return self.a * torch.cos(inner + self.offset)

    def _to_cos_sin_features(self, x, sigma=None):
        sigma = (self.sigma if sigma is None else sigma)
        coeff = self.freqs / sigma
        inner = torch.matmul(x, coeff.T)
        return self.a * torch.cat([torch.cos(inner), torch.sin(inner)], dim=-1)


class RFFKernel:
    def sample_freqs(self, shape):
        raise NotImplementedError

    def inv_cdf(self, x):
        raise NotImplementedError


class RFFKernelRBF(RFFKernel):
    def sample_freqs(self, shape):
        return np.random.normal(0.0, 1.0, shape)

    def inv_cdf(self, x):
        return erfinv(2*x-1) * np.sqrt(2)


class RFFKernelMatern12(RFFKernel):
    def sample_freqs(self, shape):
        return np.random.normal(0, 1, shape) * \
                np.sqrt(1/np.random.chisquare(1, shape))

    def inv_cdf(self, x):
        # This formula comes from the inv cdf of a standard Cauchy
        # distribution (see Laplace RFF).
        return np.tan(np.pi*(x-0.5))


class RFFKernelMatern32(RFFKernel):
    def sample_freqs(self, shape):
        return np.random.normal(0, 1, shape) * \
                np.sqrt(3/np.random.chisquare(3, shape))

    def inv_cdf(self, x):
        # From https://www.researchgate.net/publication/247441442
        # William T. Shaw. "Sampling Student’s T distribution – use of the
        # inverse cumulative distribution function".
        # Journal of Computational Finance 9(4). DOI: 10.21314/JCF.2006.150
        return (2*x - 1) / np.sqrt(2*x*(1-x))


class RFFKernelMatern52(RFFKernel):
    def sample_freqs(self, shape):
        return np.random.normal(0, 1, shape) * \
                np.sqrt(5/np.random.chisquare(5, shape))

    def inv_cdf(self, x):
        # From https://www.researchgate.net/publication/247441442
        alpha = 4*x*(1-x)
        p = 4 * np.cos(np.arccos(np.sqrt(alpha))/3) / np.sqrt(alpha)
        return np.sign(x-0.5)*np.sqrt(p-4)
