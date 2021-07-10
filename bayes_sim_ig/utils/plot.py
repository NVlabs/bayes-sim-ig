# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utils for plots"""
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from . import pdf


def plot_1d_posterior(ax, i, sim_params_names, true_params, posterior,
                      p_lower, p_upper, legend_on=False):
    minlim = p_lower[i] - 0.1 * p_lower[i]
    maxlim = p_upper[i] + 0.1 * p_upper[i]
    x_plot = np.arange(minlim, maxlim, 0.001).reshape(-1, 1)
    y_plot = posterior.eval(x_plot, ii=[i], log=False)
    p = pdf.Uniform(p_lower[i:i+1], p_upper[i:i+1])
    y_plot_prior = p.eval(x_plot, ii=None, log=False)
    ax.plot(x_plot, y_plot, '-b', label=r'Predicted posterior')
    ax.plot(x_plot, y_plot_prior, '-g', label=r'Uniform prior')
    cur_true_param = true_params.ravel()[i]
    ax.axvline(cur_true_param, c='r', label=r'True value')
    ax.axis('on')
    if legend_on:
        ax.legend(fontsize=10)
    ax.set_xlabel(sim_params_names[i], fontsize=10)
    ax.set_ylabel('likelihood', fontsize=10)


def get_2d_posterior_data(posterior, xmin=0, xmax=2, ymin=0, ymax=2,
                          nbins=100, dims=(0, 1)):
    xi, yi = np.mgrid[xmin:xmax:nbins * 1j, ymin:ymax:nbins * 1j]
    X = np.concatenate((xi.reshape(1, nbins * nbins),
                        yi.reshape(1, nbins * nbins)), axis=0)
    zi = posterior.eval(X.T, ii=dims, log=False)  # contour
    return xi, yi, zi


def plot_2d_posterior(ax, sim_params_names, true_params, posterior,
                      xmin, xmax, ymin, ymax, dims=(0, 1), data=None):
    cmap = cm.cool
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_xlabel(sim_params_names[0], fontsize=10)
    ax.set_ylabel(sim_params_names[1], fontsize=10)
    if posterior is not None:  # eval on a regular grid
        xi, yi, zi = get_2d_posterior_data(posterior, xmin=xmin, ymin=ymin,
                                           xmax=xmax, ymax=ymax, dims=dims)
    else:
        xi, yi, zi = data
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
    # ax.colorbar(spacing='proportional')
    max_lik = np.max(zi)
    true_lik = posterior.eval(
        true_params.reshape(1, -1), ii=dims, log=False, debug=False)
    # print('true_lik', true_lik)
    # true_ll = posterior.eval(
    #     true_params.reshape(1, -1), ii=dims, log=True, debug=True)
    # print('true_nll', -1.0*true_ll)
    levels = []
    if max_lik > true_lik:
        levels = np.arange(true_lik, max_lik, (max_lik-true_lik)/5.0)
    with warnings.catch_warnings():
        msg = 'No contour levels were found within the data range.'
        warnings.filterwarnings('ignore', message=msg)
        cs = ax.contour(xi, yi, zi.reshape(xi.shape), levels=levels, alpha=0.8)
    if len(levels) > 0:
        ax.clabel(cs, inline=True, fontsize=10)
    # ax.set_xticks(np.arange(xmin, xmax, 0.5), minor=True)
    # ax.set_yticks(np.arange(ymin, ymax, 0.5), minor=True)
    if true_params is not None:
        ax.scatter(true_params[0], true_params[1], 1000, 'y',
                   marker='*', label='True value')
    # Plot the component centres
    if hasattr(posterior, 'n_components'):
        xc = np.array([posterior.components[:][i].m[dims[0]]
                       for i in range(posterior.n_components)])
        yc = np.array([posterior.components[:][i].m[dims[1]]
                       for i in range(posterior.n_components)])
        ax.plot(xc, yc, 'b+', markersize=10)
    # Show grid lines.
    # ax.grid(b=True, which='minor', alpha=0.6)
    ax.grid(b=True, which='major', alpha=0.8)


def plot_posterior_pair(row, col, sim_params_names,
                        true_params, posterior, p_lower, p_upper):
    if len(true_params) == 1:
        fig, ax = plt.subplots(1, 1)
        plot_1d_posterior(ax, 0, sim_params_names, true_params,
                          posterior, p_lower, p_upper, legend_on=True)
        plt.tight_layout()
        return fig, sim_params_names[0]
    else:
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches((3*2, 3*2))
        plot_1d_posterior(axes[0,0], row, sim_params_names, true_params,
                          posterior, p_lower, p_upper, legend_on=True)
        plot_1d_posterior(axes[1,1], col, sim_params_names, true_params,
                          posterior, p_lower, p_upper, legend_on=True)
        ids = np.array([row, col])
        plot_2d_posterior(
            axes[1,0], sim_params_names[ids], true_params[ids], posterior,
            xmin=p_lower[ids[0]], ymin=p_lower[ids[1]],
            xmax=p_upper[ids[0]], ymax=p_upper[ids[1]], dims=ids)
        axes[0,1].axis('off')  # empty space
        plt.tight_layout()
        ttl = sim_params_names[row]+'_vs_'+sim_params_names[col]
    return fig, ttl


def add_fig_to_tensorboard(writer, fig, ttl, epoch):
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img / 255.0
    img = np.swapaxes(img, 0, 2)  # for TB/TF versions >= 1.8
    img = np.swapaxes(img, 1, 2)  # avoid flipped images
    writer.add_image(ttl, img, epoch)
    plt.close(fig)


def plot_posterior(writer, tb_msg, tb_step, sim_params_names, skip_ids,
                   true_params, posterior, p_lower, p_upper, output_file=None):
    matplotlib.use('Agg')  # non-interactive plot backend
    for row in range(len(true_params)):
        if row in skip_ids:
            continue
        for col in range(row+1, len(true_params)):
            if col in skip_ids:
                continue
            fig, title = plot_posterior_pair(
                row, col, sim_params_names, true_params, posterior,
                p_lower, p_upper)
            print('plotting', title)
            if writer is not None:
                add_fig_to_tensorboard(writer, fig, tb_msg+'_'+title, tb_step)
                writer.flush()
            if output_file is not None:
                plt.savefig(output_file, dpi=100)
            plt.close(fig)
