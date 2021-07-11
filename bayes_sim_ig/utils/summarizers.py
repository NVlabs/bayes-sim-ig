# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Various ways to summarize/compress trajectories."""

import gc

import torch

try:
    import signatory
finally:
    pass  # signatory is advanced experimental functionality, so not required


def pad_states_actions(states, actions, tgt_actions_len=None):
    """Ensures that we have n states and n actions for each trajectory.
    We usually collect n+1 states and n actions: s0, a1, s1, .... an, sn
    So this method pads actions to make states and actions same length.
    If tgt_actions_len is given, pads (or chops) states and actions sequences
    to ensure trajectories with length=tgt_actions_len.

    Parameters
    ----------
    states : torch.Tensor
        Trajectory states (ntraj x n_steps x state_dim)
    actions : torch.Tensor
        Trajectory actions (ntraj x n_steps x action_dim)
    tgt_actions_len: int
        Target length for the actions tensor (+1 for states tensor)

    Returns
    -------
    states : torch.Tensor
        Trajectory states, chopped/padded (ntraj x tgt_n_steps x state_dim)
    actions : torch.Tensor
        Trajectory actions chopped/padded (ntraj x tgt_n_steps x action_dim)
    """
    assert(len(states.shape) == 3), 'Need states: ntraj x n_steps x state_dim'
    assert(len(actions.shape) == 3), 'Need actions: ntraj x n_steps x state_dim'
    if tgt_actions_len is None:
        tgt_actions_len = states.shape[1]
    npad = tgt_actions_len - states.shape[1]
    if npad > 0:
        last_state = states[:, -1, :].clone()
        action_pads = last_state.repeat(1, npad, 1)
        states = torch.cat([states, action_pads], dim=1)
    else:
        states = states[:, :tgt_actions_len, :]
    npad = tgt_actions_len - actions.shape[1]
    if npad > 0:
        last_action = actions[:, -1, :].clone()
        action_pads = last_action.repeat(1, npad, 1)
        actions = torch.cat([actions, action_pads], dim=1)
    else:
        actions = actions[:, :tgt_actions_len, :]
    assert(states.shape[1] == actions.shape[1])
    return states, actions


def summary_start(states, actions, max_t=10):
    """Outputs a short initial snippet of the trajectory."""
    states, actions = pad_states_actions(states, actions, max_t)
    feats = torch.cat([states, actions], dim=-1)
    bsz = feats.shape[0]
    return feats.view(bsz, -1)


def summary_waypts(states, actions, n_waypts=10):
    """Outputs states and actions at fixed intervals to retain n_waypoints."""
    states, actions = pad_states_actions(states, actions, n_waypts)
    ntraj, traj_len, state_dim = states.shape
    assert(n_waypts <= traj_len)
    assert(traj_len == actions.shape[1])
    chunk_sz = int(traj_len/n_waypts)
    feats = torch.zeros(ntraj, n_waypts, state_dim+actions.shape[-1])
    feats = feats.to(states.device)
    traj_i = 0
    for feats_i in range(n_waypts):
        feats[:, feats_i, :state_dim] = states[:, traj_i, :]
        feats[:, feats_i, state_dim:] = actions[:, traj_i, :]
        traj_i += chunk_sz
    return feats.view(ntraj, -1)


def cross_correlation(states, actions, use_state_diff=False):
    """Cross-correlation summaries (Section IV.F of BayesSim RSS2019 paper)."""
    states, actions = pad_states_actions(states, actions)
    ntraj, traj_len, state_dim = states.shape
    assert(traj_len > 1)  # empty episodes are problematic
    assert(actions.shape[1] == traj_len)  # need traj lens same
    max_traj_len = 10     # cross corr. on long trajs is costly; so 10 waypoints
    if state_dim > 50:    # and for envs with large state dimensionality
        max_traj_len = 5  # reduce the number of waypoints we take to 5
    if traj_len > max_traj_len:
        states_actions = summary_waypts(states, actions, n_waypts=max_traj_len)
        states_actions = states_actions.view(ntraj, max_traj_len, -1)
        states = states_actions[:, :, :state_dim]
        actions = states_actions[:, :, state_dim:]
    # Make state features, concatenate with actions, mean and std stats.
    if use_state_diff:
        state_feats = states[:, :, 1:] - states[:, :, :-1]
    else:
        state_feats = states[:, :, :-1]
    state_feats = state_feats.contiguous().view(ntraj, -1)
    action_feats = actions.contiguous().view(ntraj, -1)
    # Batch outer prod https://discuss.pytorch.org/t/batch-outer-product/4025
    cross_corr = torch.bmm(state_feats.unsqueeze(2), action_feats.unsqueeze(1))
    cross_corr = cross_corr.view(ntraj, -1)
    mu = state_feats.mean(dim=-1, keepdim=True)
    if state_feats.shape[1] < 2:  # std would be nan for < 2 entries on traj
        std = torch.zeros_like(mu)
    else:
        std = state_feats.std(dim=-1, keepdim=True)
    feats = torch.cat([cross_corr, mu, std], dim=-1)
    assert(torch.isfinite(feats).all())
    print('cross_corr feats', feats.shape, feats.device)
    return feats


def summary_corrdiff(states, actions):
    return cross_correlation(states, actions, use_state_diff=True)


def summary_corr(states, actions):
    return cross_correlation(states, actions, use_state_diff=False)


def signature_depth(ndim):
    """Computes appropriate signature depth to use for path signatures.
    Note: depth>2 can run out of memory, since signatures have size ndim^depth.
    """
    max_output_dim = 110**2  # depth 2 for 110D paths
    for depth in reversed(range(4)):  # try depth 3, then 2
        if ndim**depth <= max_output_dim:
            return depth
    return 1  # depth 1 implies looking at start and end, still a signal


def summary_signatory(states, actions):
    """Computes signatures for the given batch of paths.
    Path signatures are time invariant by default. In robotics we usually do
    want to retain dependence on time (i.e. getting faster to the goal along
    the same path should be better than sitting at the start for a while, then
    sprinting to the goal. Hence, we use the common way to make path signatures
    time-dependent by extending the path with a time index.
    """
    assert(len(states.shape) == 3), 'states should be batch x time x state_dim'
    bsz, path_len, state_dim = states.shape
    time_ids = torch.arange(1, path_len+1).view(1, -1, 1).repeat(bsz, 1, 1)
    paths = torch.cat([time_ids.to(states.device), states, actions], dim=-1)
    depth = signature_depth(paths.shape[-1])
    if paths.shape[0] <= 10000:
        return signatory.signature(paths, depth=depth)
    niters = 10
    chunksz = paths.shape[0]//10
    sgns = []
    for i in range(niters):
        tmp = paths[i*chunksz:(i+1)*chunksz]
        sgn = signatory.signature(tmp, depth=depth)
        sgns.append(sgn)
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(sgns, dim=0)
