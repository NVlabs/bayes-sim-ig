# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Collect a set of simulated trajectories for training BayesSim."""

import torch
from .summarizers import *  # used dynamically


def collect_trajectories(num_trajs, ppo, summarizer_fxn, collect_policy_fxn,
                         max_traj_len=None, device='cpu',
                         verbose=False, visualize=False):
    """Collects data for num_trajs episodes from ppo.vec_env."""
    if num_trajs < ppo.vec_env.task.num_envs:
        # Pay attention only to env with ID<num_trajs. Note that waiting for
        # any first num_trajs envs to finish would bias the episodes to be
        # shorter, but using a pre-defined set of IDs to pay attention to is
        # unbiased (since performance/length does not depend on ID).
        ids = range(num_trajs)
    else:
        ids = range(ppo.vec_env.task.num_envs)
    all_sim_params = []
    all_sim_traj_summaries = []
    all_sim_traj_rewards = []
    sim_episode_obs = {}
    sim_episode_act = {}
    sim_episode_rwd = {}
    num_summarizer_failures = 0
    saved_max_episode_length = None
    if max_traj_len is not None:
        saved_max_episode_length = ppo.vec_env.task.max_episode_length
        ppo.vec_env.task.max_episode_length = max_traj_len+1
    obs = ppo.vec_env.reset()  # this overloaded method does reset all envs
    imgs = []
    # if visualize and hasattr(ppo.vec_env.task, 'get_img'):
    #     imgs = [ppo.vec_env.task.get_img(ids[0])]
    for env_id in range(ppo.vec_env.num_envs):
        sim_episode_obs[env_id] = [obs[env_id]]
        sim_episode_act[env_id] = []
        sim_episode_rwd[env_id] = []
    while True:  # collect and summarize simulated episodes/trajectories
        if hasattr(ppo.vec_env, 'get_state'):
            act, *_ = ppo.actor_critic.act(obs, ppo.vec_env.get_state())
        else:
            act, *_ = ppo.actor_critic.act(obs)  # act in [-1, 1]
        if collect_policy_fxn is not None:
            act = collect_policy_fxn(act)  # appropriate act (also in [-1, 1])
        obs, rwd, done, _ = ppo.vec_env.step(act)
        for env_id in ids:
            sim_episode_act[env_id].append(act[env_id].clone())
            if len(sim_episode_act) > 1:  # 0th act is from previous traj
                sim_episode_rwd[env_id].append(rwd[env_id].clone())
                sim_episode_obs[env_id].append(obs[env_id].clone())
            if done[env_id] == 1:  # will be reset on next step, so save traj
                assert(len(sim_episode_obs[env_id]) <=
                       ppo.vec_env.task.max_episode_length)
                if summarizer_fxn is None:
                    sim_x = None
                elif '_batch' in summarizer_fxn.__name__:  # summarize later
                    sim_x = summary_start(sim_episode_obs[env_id],
                                          sim_episode_act[env_id],
                                          ppo.vec_env.task.max_episode_length)
                else:
                    sim_x = summarizer_fxn(sim_episode_obs[env_id],
                                           sim_episode_act[env_id])
                if (summarizer_fxn is None) or (sim_x is not None):
                    all_sim_params.append(torch.tensor(
                        ppo.vec_env.task.extern_actor_params[env_id]).float())
                    all_sim_traj_summaries.append(sim_x)
                    all_sim_traj_rewards.append(sum(sim_episode_rwd[env_id]))
                    num_trajs_done = len(all_sim_traj_summaries)
                    if (verbose and num_trajs > 10 and
                        num_trajs_done%(num_trajs//10) == 0):
                        print('collected', len(all_sim_traj_summaries), 'trajs')
                    if len(all_sim_traj_summaries) >= num_trajs:
                        break  # collected enough trajectories (episodes)
                else:
                    num_summarizer_failures += 1
                # Clear episode accumulators.
                sim_episode_obs[env_id] = []
                sim_episode_act[env_id] = []
                sim_episode_rwd[env_id] = []
        if (visualize and hasattr(ppo.vec_env.task, 'get_img') and
            len(imgs) < ppo.vec_env.task.max_episode_length):
            imgs.append(ppo.vec_env.task.get_img())
        if num_summarizer_failures > num_trajs*10:
            assert(False), 'ERROR: too many summarizer failures'
        if len(all_sim_traj_summaries) >= num_trajs:
            break  # collected enough trajectories (episodes)
    sim_params_smpls = torch.stack(all_sim_params).to(device)
    sim_traj_rewards = torch.stack(all_sim_traj_rewards).to(device)
    sim_traj_summaries = None
    if summarizer_fxn is not None:
        sim_traj_summaries = torch.stack(all_sim_traj_summaries).to(device)
    if summarizer_fxn is not None and '_batch' in summarizer_fxn.__name__:
        bsz = sim_traj_summaries.shape[0]
        sim_traj_summaries = sim_traj_summaries.view(
            bsz, ppo.vec_env.task.max_episode_length, -1)
        sim_traj_summaries = summarizer_fxn(sim_traj_summaries)
    if saved_max_episode_length is not None:
        ppo.vec_env.task.max_episode_length = saved_max_episode_length
    return sim_params_smpls, sim_traj_summaries, sim_traj_rewards, imgs


def policy_ones(act):
    return torch.ones_like(act)


def policy_random(act):
    return torch.rand_like(act)


def policy_rl(act):
    return act


def policy_rl_randomized(act, frac_rnd=0.1):
    rnd = torch.rand((1,)).item()
    if rnd < frac_rnd:
        act = torch.rand_like(act)*2 - 1.0  # U[-1,1]
    return act

