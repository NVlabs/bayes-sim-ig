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


def ensure_torch_tensor(item):
    if isinstance(item, list):
        if len(item) == 0:
            return None
        item = torch.stack(item)
    assert(torch.isfinite(item).all())
    return item


def pad_states_actions(states, actions, tgt_actions_len=None):
    """Ensures that we have n+1 states and n actions.
    Pads states and actions sequences to obtain the trajectory with
    length=tgt_actions_len, if needed.
    """
    states = ensure_torch_tensor(states)
    actions = ensure_torch_tensor(actions)
    if states.shape[0] == actions.shape[0]:
        states = torch.cat([states, states[-1:,:]], dim=0)  # repeat last state
    assert(states.shape[0]-1 == actions.shape[0])   # s0, a1, s1, .... an, sn
    if (tgt_actions_len is not None) and (tgt_actions_len-actions.shape[0]) > 0:
        npad = tgt_actions_len - actions.shape[0]
        spads = torch.ones(npad, *list(states.shape[1:])).to(states.device)
        spads[:, :] = states[-1, :]  # pad with the last state
        states = torch.cat([states, spads], dim=0)
        apads = torch.ones(npad, *list(actions.shape[1:])).to(states.device)
        apads[:, :] = actions[-1, :]  # pad with the last action
        actions = torch.cat([actions, apads], dim=0)
    return states, actions


def summary_start(states, actions, max_t=10):
    """Outputs a short initial snippet of the trajectory."""
    states, actions = pad_states_actions(states, actions, max_t)
    feats = torch.cat([states[:max_t], actions[:max_t]], dim=-1)
    return feats.view(-1)


def summary_waypts(states, actions, n_waypts=10):
    """Outputs states and actions at fixed intervals to retain n_waypoints."""
    states, actions = pad_states_actions(states, actions, n_waypts)
    assert(n_waypts <= actions.shape[0])
    chunk_sz = int(states.shape[0]/n_waypts)
    feats = torch.zeros(n_waypts, states.shape[1]+actions.shape[1])
    traj_i = 0
    for feats_i in range(n_waypts):
        feats[feats_i, :states.shape[1]] = states[traj_i, :]
        feats[feats_i, states.shape[1]:] = actions[traj_i, :]
        traj_i += chunk_sz
    return feats.view(-1)


def cross_correlation(states, actions, use_state_diff=False, use_actions=True):
    """Cross-correlation summaries (Section IV.F of BayesSim RSS2019 paper)."""
    states, actions = pad_states_actions(states, actions)
    min_episode_len = 2  # discard very short episodes
    if states is None or actions is None or states.shape[0] < min_episode_len:
        return None
    # Make state features, concatenate with actions, mean and std stats.
    if use_state_diff:
        state_feats = states[1:] - states[:-1]
    else:
        state_feats = states[:-1]
    assert(state_feats.shape[0] == actions.shape[0])
    cross_corr = torch.matmul(state_feats.t(), actions)
    mu = state_feats.mean(dim=0)
    if state_feats.shape[0] < 2:  # std would be nan for < 2 entries
        std = torch.zeros_like(mu)
    else:
        std = state_feats.std(dim=0)
    feats = torch.cat([cross_corr.view(-1), mu, std], dim=-1)
    assert(torch.isfinite(feats).all())
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
    """Computes a signature for the given path.
    Note: signatory has an init overhead, so this function should be used only
    for debugging. Use summary_signatory_batch as the summarizer function
    """
    states, actions = pad_states_actions(states, actions)
    assert(states.shape[0]-1 == actions.shape[0])
    # Signatory expects batch x timesteps x dims
    path = torch.cat([states[:-1], actions], dim=-1).unsqueeze(0)
    return summary_signatory_batch(path)


def summary_signatory_batch(paths):
    """Computes signatures for the given batch of paths.
    Path signatures are time invariant by default. In robotics we usually do
    want to retain dependence on time (i.e. getting faster to the goal along
    the same path should be better than sitting at the start for a while, then
    sprinting to the goal. Hence, we use the common way to make path signatures
    time-dependent by extending the path with a time index.
    """
    assert(len(paths.shape) == 3), 'paths should be batch x time x dim'
    bsz, path_len, ndim = paths.shape
    time_ids = torch.arange(1, path_len+1).view(1, -1, 1).repeat(bsz, 1, 1)
    paths = torch.cat([time_ids.to(paths.device), paths], dim=-1)
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
