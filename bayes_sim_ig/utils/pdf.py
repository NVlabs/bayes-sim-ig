# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Definitions and utilities for Uniform, Gaussian and Mixture of Gaussians.

Note: the following utilizes numpy and scipy libraries instead of pytorch.
Speed is not a major concern, since this functionality is only used for
computing BayesSim posterior after training NN models is completed.

The code below derives from:
https://github.com/gpapamak/epsilon_free_inference/blob/master/util/pdf.py
The license for the above source is pasted below:

Copyright (c) 2016, George Papamakarios
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of anybody else.

"""

import numpy as np
import numpy.random as rng
import scipy
from scipy.stats import norm
from scipy.special import erfinv
from scipy.special import logsumexp  # was in scipy.misc until 1.1.0
import ghalton

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


def discrete_sample(p, n_samples=1):
    """Samples from a discrete distribution.

    Parameters
    ----------
    p: a distribution with N elements
    n_samples: number of samples

    Returns
    ----------
    res: vector of samples
    """
    cumul_distr = np.cumsum(p[:-1])[np.newaxis, :]
    rnd = rng.rand(n_samples, 1)
    res = np.sum((rnd > cumul_distr).astype(int), axis=1)
    return res


class Uniform:
    """Implements a Uniform pdf.

    Parameters
    ----------
    lb_array: numpy.array
      Lower bounds
    ub_array: numpy.array
      Upper bounds
    """
    def __init__(self, lb_array=None, ub_array=None):
        self.lb_array = lb_array
        self.ub_array = ub_array
        self.cur_index = 0
        self.cached_samples = None
        assert len(lb_array) == len(ub_array)
        self.param_dim = len(lb_array)

    def __str__(self):
        """Makes a verbose string representation for debug printing."""
        res = 'Uniform: \nlower bounds:\n' + str(self.lb_array) + \
              '\nupper bounds:\n' + str(self.ub_array)
        return res

    def generate_halton_samples(self, n_samples=1000):
        """Generates Halton samples.

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate

        Returns
        ----------
        h_sample: numpy.array
          A vector of samples
        """
        domain = np.zeros((2, len(self.ub_array)))
        for ix in range(self.param_dim):
            domain[0][ix] = self.lb_array[0]
            domain[1][ix] = self.ub_array[1]
        dim = domain.shape[1]
        perms = ghalton.EA_PERMS[:dim]
        sequencer = ghalton.GeneralizedHalton(perms)
        h_sample = np.array(sequencer.get(n_samples + 1))[1:]
        if dim == 1:
            h_sample = domain[0] + h_sample * (domain[1] - domain[0])
        else:
            h_sample = domain[0, :] + h_sample * (domain[1, :] - domain[0, :])
        return h_sample

    def gen(self, n_samples=1, method='random'):
        """Generates samples.

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate
        method: string, optional; 'random' or 'halton'
            Use Halton sampling if 'halton', random uniform if 'random'.

        Returns
        ----------
        result: numpy.array
          A vector of samples
        """
        result = None
        if method == 'halton':
            result = self.generate_halton_samples(n_samples=n_samples)
        elif method == 'random':
            for ix in range(len(self.lb_array)):
                samples = np.random.uniform(self.lb_array[ix],
                                            self.ub_array[ix], size=n_samples)
                if result is None:
                    result = samples
                else:
                    result = np.concatenate((result, samples), axis=0)
        else:
            raise ValueError('Unknown gen method '+method)
        return result.reshape(-1, len(self.lb_array))

    def eval(self, x, ii=None, log=True, debug=False):
        """Evaluates Uniform PDF

        Parameters
        ----------
        x : int or list or np.array
            Rows are inputs to evaluate at
        ii : list
            A list of indices specifying which marginal to evaluate.
            If None, the joint pdf is evaluated
        log : bool, defaulting to True
            If True, the log pdf is evaluated

        Returns
        -------
        p: float
          PDF or log PDF
        """
        if ii is None:
            ii = np.arange(self.param_dim)
        N = np.atleast_2d(x).shape[0]
        p = 1/np.prod(self.ub_array[ii] - self.lb_array[ii])
        p = p*np.ones((N,))  # broadcasting
        # truncation of density
        ind = (x > self.lb_array[ii]) & (x < self.ub_array[ii])
        p[np.prod(ind,axis=1) == 0] = 0
        if log:
            if not ind.any():
                raise ValueError('log prob. not defined outside of truncation')
            else:
                return np.log(p)
        else:
            return p


class Gaussian:
    """Implements a Gaussian pdf. Focus is on efficient multiplication,
       division and sampling."""
    def __init__(self, m=None, P=None, U=None, S=None, Pm=None, L=None):
        """
        Initializes a Gaussian pdf given a valid combination of its parameters.
        Valid combinations are: m-P, m-U, m-S, Pm-P, Pm-U, Pm-S.

        Parameters
        ----------
        m: numpy.array
            Mean
        P: numpy.array
             Precision
        U: numpy.array
             Upper triangular precision factor (U'U = P)
        S: numpy.array
             Covariance matrix
        C : numpy.array
             Upper or lower triangular covariance factor (S = C'C)
        Pm: numpy.array
             Precision times mean such that P*m = Pm
        L: numpy.array
            Lower triangular covariance factor given as 1D array (LL' = S)
        """
        if m is not None:
            m = np.asarray(m)
            self.m = m
            self.ndim = m.size
            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(self.P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif L is not None:
                L = np.asarray(L)
                Lm = np.diag(L[0:self.ndim])
                if 1 < self.ndim < L.shape[0]:  # if full covariance
                    tril_ids = np.tril_indices(self.ndim, -1)
                    Lm[tril_ids[0], tril_ids[1]] = L[self.ndim:]
                self.C = Lm.T
                self.S = np.dot(self.C.T, self.C)
                self.P = np.linalg.inv(self.S)
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))
            else:
                raise ValueError('Precision information missing.')
        elif Pm is not None:
            Pm = np.asarray(Pm)
            self.Pm = Pm
            self.ndim = Pm.size
            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(self.P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.m = np.dot(S, Pm)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing.')
        else:
            raise ValueError('Mean information missing.')

    def gen(self, n_samples=1, method='random'):
        """Returns independent samples from the Gaussian."""
        if method == 'random':
            z = rng.randn(n_samples, self.ndim)
            samples = np.dot(z, self.C) + self.m
        elif method == 'halton':
            perms = ghalton.EA_PERMS[:self.ndim]
            sequencer = ghalton.GeneralizedHalton(perms)
            samples = np.array(sequencer.get(int(n_samples) + 1))[1:]
            z = erfinv(2 * samples - 1) * np.sqrt(2)
            samples = np.dot(z, self.C) + self.m
        else:
            raise ValueError('Unknown gen method '+method)
        return samples

    def eval(self, x, ii=None, log=True):
        """
        Evaluates the Gaussian pdf.

        Parameters
        ----------
        x: numpy.array
            input data (rows are inputs to evaluate at)
        ii: list
            A list of indices specifying which marginal to evaluate;
            if None, the joint pdf is evaluated
        log: bool
            if True, the log pdf is evaluated

        Returns
        ----------
        res: float
          PDF or log PDF
        """
        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5
        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            eps = 1.e-5*S.mean()*np.diag(np.random.rand(S.shape[0]))
            lp = scipy.stats.multivariate_normal.logpdf(x, m, S+eps)
            lp = np.array([lp]) if x.shape[0] == 1 else lp
        res = lp if log else np.exp(lp)
        return res

    def __mul__(self, other):
        """Multiply with another Gaussian."""
        assert isinstance(other, Gaussian)
        P = self.P + other.P
        Pm = self.Pm + other.Pm
        return Gaussian(P=P, Pm=Pm)

    def __imul__(self, other):
        """Incrementally multiply with another Gaussian."""
        assert isinstance(other, Gaussian)
        res = self * other
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def __div__(self, other):
        """Divide by another Gaussian.
           The resulting Gaussian might be improper."""
        assert isinstance(other, Gaussian)
        P = self.P - other.P
        Pm = self.Pm - other.Pm
        return Gaussian(P=P, Pm=Pm)

    def __idiv__(self, other):
        """Incrementally divide by another Gaussian.
           The resulting Gaussian might be improper."""
        assert isinstance(other, Gaussian)
        res = self / other
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def __pow__(self, power, modulo=None):
        """Raise Gaussian to a power and get another Gaussian."""
        P = power * self.P
        Pm = power * self.Pm
        return Gaussian(P=P, Pm=Pm)

    def __ipow__(self, power):
        """Incrementally raise Gaussian to a power."""
        res = self ** power
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def kl(self, other):
        """Calculates the kl divergence from this to another Gaussian,
           i.e. KL(this | other)."""
        assert isinstance(other, Gaussian)
        assert self.ndim == other.ndim
        t1 = np.sum(other.P * self.S)
        m = other.m - self.m
        t2 = np.dot(m, np.dot(other.P, m))
        t3 = self.logdetP - other.logdetP
        t = 0.5 * (t1 + t2 + t3 - self.ndim)
        return t


class MoG:
    """Implements a mixture of Gaussians."""
    def __init__(self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None, Ls=None):
        """
        Creates a mog with a valid combination of parameters.

        Parameters
        ----------
        a: list or numpy.array
            Mixing coefficients (mixture weights).
        ms: numpy.array
            Component means.
        Ps: numpy.array
            Precisions
        Us: numpy.array
            Precision factors (U'U = P)
        Ss: numpy.array
            Covariances
        xs: list or numpy.array
            Gaussian variables
        Ls: numpy.array
            Lower-triangular covariance factor (L*L' = S)
        """
        if ms is not None:
            if Ps is not None:
                self.xs = [Gaussian(m=m, P=P) for m, P in zip(ms, Ps)]
            elif Us is not None:
                self.xs = [Gaussian(m=m, U=U) for m, U in zip(ms, Us)]
            elif Ss is not None:
                self.xs = [Gaussian(m=m, S=S) for m, S in zip(ms, Ss)]
            elif Ls is not None:
                self.xs = [Gaussian(m=m, L=L) for m, L in zip(ms, Ls)]
            else:
                raise ValueError('Precision information missing.')
        elif xs is not None:
            self.xs = xs
        else:
            raise ValueError('Mean information missing.')
        self.a = np.asarray(a)
        self.ndim = self.xs[0].ndim
        self.n_components = len(self.xs)
        self.ncomp = self.n_components

    @property
    def weights(self):
        return self.a

    @property
    def components(self):
        return self.xs

    def gen(self, n_samples=1, method='random'):
        """Generates independent samples from mog."""
        ii = discrete_sample(self.a, n_samples)
        ns = [np.sum((ii == i).astype(int)) for i in range(self.n_components)]
        samples = [x.gen(n_samples=n, method=method)
                   for x, n in zip(self.xs, ns)]
        samples = np.concatenate(samples, axis=0)
        return samples

    def eval(self, x, ii=None, log=True, debug=False):
        """
        Evaluates the mog pdf.
        x: rows are inputs to evaluate at
        ii: a list of indices specifying which marginal to evaluate;
                   if None, the joint pdf is evaluated
        log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """
        ps = np.array([self.xs[ix].eval(x, ii, log)
                       for ix in range(len(self.a))]).T
        if log:
            res = scipy.special.logsumexp(ps + np.log(self.a), axis=1)
        else:
            res = np.dot(ps, self.a)
        if debug:
            print('weights\n', self.a, '\nps\n', ps, '\nres\n', res)
        return res

    def __str__(self):
        """Makes a verbose string representation for debug printing."""
        mus = np.array([gauss.m.tolist() for gauss in self.xs])
        diagS = np.array([np.diagonal(gauss.S).tolist() for gauss in self.xs])
        res = 'MoG:\nweights:\n' + str(self.a) + '\nmeans:\n' + str(mus) + \
              '\ndiagS:\n' + str(diagS)
        return res

    def __mul__(self, other):
        """Multiplies by a single Gaussian."""
        assert isinstance(other, Gaussian)
        ys = [x * other for x in self.xs]
        lcs = np.empty_like(self.a)
        for i, (x, y) in enumerate(zip(self.xs, ys)):
            lcs[i] = x.logdetP + other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m))
            lcs[i] += np.dot(other.m, np.dot(other.P, other.m))
            lcs[i] -= np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5
        la = np.log(self.a) + lcs
        la -= logsumexp(la)
        a = np.exp(la)
        return MoG(a=a, xs=ys)

    def __imul__(self, other):
        """Incrementally multiplies by a single Gaussian."""
        assert isinstance(other, Gaussian)
        res = self * other
        self.a = res.a
        self.xs = res.xs
        return res

    def __div__(self, other):
        """Divides by a single Gaussian."""
        assert isinstance(other, Gaussian)
        ys = [x / other for x in self.xs]
        lcs = np.empty_like(self.a)
        for i, (x, y) in enumerate(zip(self.xs, ys)):
            lcs[i] = x.logdetP - other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m))
            lcs[i] -= np.dot(other.m, np.dot(other.P, other.m))
            lcs[i] -= np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5
        la = np.log(self.a) + lcs
        la -= logsumexp(la)
        a = np.exp(la)
        return MoG(a=a, xs=ys)

    def __idiv__(self, other):
        """Incrementally divides by a single Gaussian."""
        assert isinstance(other, Gaussian)
        res = self / other
        self.a = res.a
        self.xs = res.xs
        return res

    def calc_mean_and_cov(self):
        """Calculates the mean vector and the covariance matrix of the MoG."""
        ms = [x.m for x in self.xs]
        m = np.dot(self.a, np.array(ms)[np.newaxis, :])
        Ss = [x.sigma for x in self.xs]
        S = np.dot(self.a, np.array(Ss)[np.newaxis, :])
        return m, S

    def project_to_gaussian(self):
        """Returns a Gaussian with the same mean and precision as the MoG."""
        m, S = self.calc_mean_and_cov()
        return Gaussian(m=m, S=S)

    def prune_negligible_components(self, threshold):
        """Removes all components with mixing coefficients < threshold."""
        ii = np.nonzero((self.a < threshold).astype(int))[0]
        total_del_a = np.sum(self.a[ii])
        del_count = ii.size
        self.n_components -= del_count
        self.a = np.delete(self.a, ii)
        self.a += total_del_a / self.n_components
        self.xs = [x for i, x in enumerate(self.xs) if i not in ii]

    def kl(self, other, n_samples=10000):
        """Estimates the kl from this to another pdf,
           i.e. KL(this | other), using Monte Carlo."""
        x = self.gen(n_samples)
        lp = self.eval(x, log=True)
        lq = other.eval(x, log=True)
        t = lp - lq
        res = np.mean(t)
        err = np.std(t, ddof=1) / np.sqrt(n_samples)
        return res, err


def fit_mog(x, n_components, w=None, tol=1.0e-9, maxiter=float('inf'),
            verbose=False):
    """Fits a mixture of Gaussians to (possibly weighted) data using
    expectation maximization."""
    x = x[:, np.newaxis] if x.ndim == 1 else x
    n_data, n_dim = x.shape
    #
    # Initialize.
    a = np.ones(n_components) / n_components
    ms = rng.randn(n_components, n_dim)
    Ss = [np.eye(n_dim) for _ in range(n_components)]
    iter = 0
    #
    # Calculate log p(x,z), log p(x) and total log likelihood.
    logPxz = np.array([scipy.stats.multivariate_normal.logpdf(
        x, ms[k], Ss[k]) for k in range(n_components)])
    logPxz += np.log(a)[:, np.newaxis]
    logPx = logsumexp(logPxz, axis=0)
    loglik_prev = np.mean(logPx) if w is None else np.dot(w, logPx)

    while True:
        #
        # E step
        z = np.exp(logPxz - logPx)
        #
        # M step
        if w is None:
            Nk = np.sum(z, axis=1)
            a = Nk / n_data
            ms = np.dot(z, x) / Nk[:, np.newaxis]
            for k in range(n_components):
                xm = x - ms[k]
                Ss[k] = np.dot(xm.T * z[k], xm) / Nk[k]
        else:
            zw = z * w
            a = np.sum(zw, axis=1)
            ms = np.dot(zw, x) / a[:, np.newaxis]
            for k in range(n_components):
                xm = x - ms[k]
                Ss[k] = np.dot(xm.T * zw[k], xm) / a[k]
        #
        # Calculate log p(x,z), log p(x) and total log likelihood.
        logPxz = np.array([scipy.stats.multivariate_normal.logpdf(
            x, ms[k], Ss[k], allow_singular=True) for k in range(n_components)])
        logPxz += np.log(a)[:, np.newaxis]
        logPx = logsumexp(logPxz, axis=0)
        loglik = np.mean(logPx) if w is None else np.dot(w, logPx)
        #
        # Check progress.
        iter += 1
        diff = loglik - loglik_prev
        # assert diff >= 0.0, 'Log likelihood decreased! There is a bug!'
        if verbose:
            print('Iteration = {0}, log likelihood = {1}, diff = {2}'.format(
                iter, loglik, diff))
        if diff < tol or iter > maxiter: break
        loglik_prev = loglik

    return MoG(a=a, ms=ms, Ss=Ss)
