# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Collect a set of simulated trajectories for training BayesSim."""

import torch

from .summarizers import pad_states_actions


def collect_trajectories(num_trajs, ppo, collect_policy_fxn,
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
    all_sim_traj_states = []
    all_sim_traj_actions = []
    all_sim_traj_rewards = []
    sim_episode_obs = {}
    sim_episode_act = {}
    sim_episode_rwd = {}
    saved_max_episode_length = None
    if max_traj_len is not None:
        saved_max_episode_length = ppo.vec_env.task.max_episode_length
        ppo.vec_env.task.max_episode_length = max_traj_len+1
    obs = ppo.vec_env.reset()  # this overloaded method does reset all envs
    imgs = []
    if visualize and hasattr(ppo.vec_env.task, 'get_img'):
        imgs = [ppo.vec_env.task.get_img()]
    for env_id in range(ppo.vec_env.num_envs):
        sim_episode_obs[env_id] = [obs[env_id]]
        sim_episode_act[env_id] = []
        sim_episode_rwd[env_id] = []
    while True:  # collect simulated episodes/trajectories
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
                all_sim_params.append(torch.tensor(
                    ppo.vec_env.task.extern_actor_params[env_id]).float())
                traj_states, traj_actions = pad_states_actions(
                    torch.stack(sim_episode_obs[env_id]).unsqueeze(0),
                    torch.stack(sim_episode_act[env_id]).unsqueeze(0),
                    tgt_actions_len=ppo.vec_env.task.max_episode_length)
                all_sim_traj_states.append(traj_states)
                all_sim_traj_actions.append(traj_actions)
                all_sim_traj_rewards.append(sum(sim_episode_rwd[env_id]))
                num_trajs_done = len(all_sim_traj_rewards)
                if (verbose and num_trajs > 10 and
                    num_trajs_done%(num_trajs//10) == 0):
                    print('collected', num_trajs_done, 'trajs')
                if num_trajs_done >= num_trajs:
                    break  # collected enough trajectories (episodes)
                # Clear episode accumulators.
                sim_episode_obs[env_id] = []
                sim_episode_act[env_id] = []
                sim_episode_rwd[env_id] = []
        if (visualize and hasattr(ppo.vec_env.task, 'get_img') and
            len(imgs) < ppo.vec_env.task.max_episode_length):
            imgs.append(ppo.vec_env.task.get_img())
        if len(all_sim_traj_rewards) >= num_trajs:
            break  # collected enough trajectories (episodes)
    sim_params_smpls = torch.stack(all_sim_params).to(device)
    sim_traj_states = torch.cat(all_sim_traj_states, dim=0).to(device)
    sim_traj_actions = torch.cat(all_sim_traj_actions, dim=0).to(device)
    sim_traj_rewards = torch.stack(all_sim_traj_rewards).to(device)
    if saved_max_episode_length is not None:
        ppo.vec_env.task.max_episode_length = saved_max_episode_length
    return sim_params_smpls, sim_traj_states, sim_traj_actions, \
           sim_traj_rewards, imgs


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

