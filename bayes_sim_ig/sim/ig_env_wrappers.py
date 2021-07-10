# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Wrappers for IG tasks to enable domain randomization and full resets."""

import os

import torch

from rlgpu.tasks.base.vec_task import VecTaskPython
from rlgpu.tasks.ant import Ant
from rlgpu.tasks.anymal import Anymal
from rlgpu.tasks.cartpole import Cartpole
from rlgpu.tasks.ball_balance import BallBalance
from rlgpu.tasks.franka import FrankaCabinet
from rlgpu.tasks.humanoid import Humanoid
from rlgpu.tasks.ingenuity import Ingenuity
from rlgpu.tasks.quadcopter import Quadcopter
from rlgpu.tasks.shadow_hand import ShadowHand
from rlgpu.utils.config import parse_sim_params

from .apply_randomizations import CustomRandomizer
from .openai_env_wrappers import PendulumB  # used dynamically
from .params_generator import ParamsGenerator

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


class AntB(Ant):
    """BayesSim-enabled version of Ant."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        Ant.__init__(self, cfg, sim_params, physics_engine,
                     device_type, device_id, headless)
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            # plot: torso, front_right_leg, left_back_foot, ankle_1, hip_1
            plot_names_skip_patterns=[
                # 'torso'
                'torso_1', 'torso_2', 'torso_3', 'torso_4',
                # 'front_right_leg',
                'front_left_leg', 'right_back_leg', 'left_back_leg',
                'front_right_foot', 'front_left_foot', 'right_back_foot',
                # 'left_back_foot',
                # 'ankle_1',
                'ankle_2', 'ankle_3', 'ankle_4',
                # 'hip_1',
                'hip_2', 'hip_3', 'hip_4',
            ])


class AnymalB(Anymal):
    """BayesSim-enabled version of Anymal."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        # IG code did not update Anymal code to load from a custom path.
        # So users have to run this code from isaacgym/python/rlgpu directory.
        # TODO: Ask IG team to fix this in the main IG code.
        asset_file = cfg['env']['asset']['assetFileName']
        asset_path = os.path.join('../../assets', asset_file)
        print('Looking for', asset_path)
        if not os.path.exists(asset_path):
            assert(False), 'Please run from isaacgym/python/rlgpu directory'
        Anymal.__init__(self, cfg, sim_params, physics_engine,
                        device_type, device_id, headless)
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            # plot: base, LF_HIP, LH_THIGH, RF_SHANK, RH_SHANK
            plot_names_skip_patterns=[
                # 'base', 'LF_HIP',
                'LF_THIGH', 'LF_SHANK', 'LH_HIP',
                # 'LH_THIGH',
                'LH_SHANK', 'RF_HIP', 'RF_THIGH',
                # 'RF_SHANK',
                'RH_HIP', 'RH_THIGH',
                # 'RH_SHANK',
            ])

    def post_physics_step(self):
        self.randomize_buf += 1
        super().post_physics_step()

    def reset(self, env_ids):
        super().reset(env_ids)
        # IG udercounts 2 steps in compute_hand_reward:
        # time_out = episode_lengths > max_episode_length
        # other envs have something like:
        # reset = torch.where(progress_buf >= max_episode_length - 1,
        # TODO: Ask IG team to make termination consistent.
        self.progress_buf[env_ids] = 2  # IG udercounts 2 steps for this task
        self.reset_buf[env_ids] = 0  # not clear why this is set to 1 in IG


class CartpoleB(Cartpole):
    """BayesSim-enabled version of Cartpole."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        Cartpole.__init__(self, cfg, sim_params, physics_engine,
                          device_type, device_id, headless)
        self.randomization_params = self.cfg['task']['randomization_params']
        self.randomize = self.cfg['task']['randomize']
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            plot_names_skip_patterns=['slider'])

    def reset(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        super().reset(env_ids)

    def post_physics_step(self):
        self.randomize_buf += 1
        super().post_physics_step()


class BallBalanceB(BallBalance):
    """BayesSim-enabled version of BallBalance."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        BallBalance.__init__(self, cfg, sim_params, physics_engine,
                             device_type, device_id, headless)
        self.randomization_params = self.cfg['task']['randomization_params']
        self.randomize = self.cfg['task']['randomize']
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            # plot: sphere, tray, upper_leg0, lower_leg1,
            # upper_leg_join0, lower_leg_joint2
            plot_names_skip_patterns=[
                # 'sphere', tray, upper_leg0,
                'upper_leg0', 'lower_leg0', 'upper_leg1',
                # 'lower_leg1',
                'upper_leg2', 'lower_leg2',
                # upper_leg_joint0,
                'upper_leg_joint1', 'upper_leg_joint2',
                'lower_leg_joint0', 'lower_leg_joint1',
                # 'lower_leg_joint2',
            ])

    def reset(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        super().reset(env_ids)

    def post_physics_step(self):
        self.randomize_buf += 1
        super().post_physics_step()


class FrankaCabinetB(FrankaCabinet):
    """BayesSim-enabled version of FrankaCabinet."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        self.randomize = False  # tmp val, reset() called in init()
        FrankaCabinet.__init__(self, cfg, sim_params, physics_engine,
                               device_type, device_id, headless)
        self.randomization_params = self.cfg['task']['randomization_params']
        self.randomize = self.cfg['task']['randomize']
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            # plot: panda_link0, panda_leftfinger,
            # panda_joint0, panda_finger_joint1, door_right,
            plot_names_skip_patterns=[
                # 'panda_link0',
                'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4',
                'panda_link5', 'panda_link6', 'panda_link7',
                # 'panda_leftfinger',
                'panda_rightfinger', 'door_left',
                # 'door_right',
                'drawer_bottom', 'drawer_top'
                # 'panda_joint0',
                'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7',
                # 'panda_finger_joint1',
                'panda_finger_joint2'
            ])

    def reset(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        super().reset(env_ids)

    def post_physics_step(self):
        self.randomize_buf += 1
        super().post_physics_step()


class HumanoidB(Humanoid):
    """BayesSim-enabled version of Humanoid."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        Humanoid.__init__(self, cfg, sim_params, physics_engine,
                          device_type, device_id, headless)
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            # plot: torso, pelvis, right_hip_x, left_knee
            plot_names_skip_patterns=[
                # 'torso',
                'head', 'torso_1', 'lower_waist',
                # 'pelvis',
                'right_thigh', 'right_shin', 'right_foot',
                'left_thigh', 'left_shin', 'left_foot',
                'right_upper_arm', 'right_lower_arm', 'right_hand',
                'left_upper_arm', 'left_lower_arm', 'left_hand',
                'abdomen_x', 'abdomen_y', 'abdomen_z',
                'right_hip_x', 'right_hip_y',
                # 'right_hip_z',
                'right_knee', 'right_ankle',
                'right_shoulder1', 'right_shoulder2', 'right_elbow',
                'left_hip_x', 'left_hip_y', 'left_hip_z',
                # 'left_knee',
                'left_ankle_y', 'left_ankle_x',
                'left_shoulder1', 'left_shoulder2', 'left_elbow',
            ])


class IngenuityB(Ingenuity):
    """BayesSim-enabled version of Ingenuity."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        # IG code did not update Ingenuity code to load from a custom path.
        # So users have to run this code from isaacgym/python/rlgpu directory.
        # TODO: Ask IG team to fix this in the main IG code.
        asset_file = cfg['env']['asset']['assetFileName']
        asset_path = os.path.join('../../assets', asset_file)
        print('Looking for', asset_path)
        if not os.path.exists(asset_path):
            assert(False), 'Please run from isaacgym/python/rlgpu directory'
        Ingenuity.__init__(self, cfg, sim_params, physics_engine,
                           device_type, device_id, headless)
        self.randomization_params = self.cfg['task']['randomization_params']
        self.randomize = self.cfg['task']['randomize']
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            # plot: chassis, rotor, rotor_bottom_roll1
            plot_names_skip_patterns=[
                # 'chassis', 'rotor',
                'rotor_physics_0', 'rotor_visual_0', 'rotor_visual_1',
                'rotor_top_roll0', 'rotor_bottom_roll0', 'rotor_top_roll1',
                # rotor_bottom_roll1
            ])
        # Make DoF names unique.
        # TODO: Ask IG team to fix this in the main IG code.
        replaced = [False, False]
        for i in range(len(self.actor_params_generator._names)):
            nm = self.actor_params_generator._names[i]
            for j in range(2):
                pfx = 'rotor_roll'+str(j)
                if nm.startswith(pfx):
                    parts = nm.split(pfx)
                    assert(len(parts)==2)
                    mid = 'one' if not replaced[j] else 'two'
                    new_pfx = 'rotor_'+mid+'_roll'+str(j)
                    new_nm = new_pfx+parts[1]
                    self.actor_params_generator._names[i] = new_nm
                    replaced[j] = True

    def reset(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        return super().reset(env_ids)  # IG code for this task expects return

    def post_physics_step(self):
        self.randomize_buf += 1
        super().post_physics_step()


class QuadcopterB(Quadcopter):
    """BayesSim-enabled version of Quadcopter."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        # IG code did not update Quadcopter code to load from a custom path.
        # So users have to run this code from isaacgym/python/rlgpu directory.
        # TODO: Ask IG team to fix this in the main IG code.
        asset_file = cfg['env']['asset']['assetFileName']
        print('Looking for', asset_file)
        if not os.path.exists(asset_file):
            assert(False), 'Please run from isaacgym/python/rlgpu directory'
        Quadcopter.__init__(self, cfg, sim_params, physics_engine,
                            device_type, device_id, headless)
        self.randomization_params = self.cfg['task']['randomization_params']
        self.randomize = self.cfg['task']['randomize']
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            # plot: chassis, rotor0, rotor1
            plot_names_skip_patterns=[
                # 'chassis', 'rotor0', 'rotor1', 'rotor_arm3'
                'rotor2', 'rotor3',
                'rotor_arm0', 'rotor_arm1', 'rotor_arm2',
                # 'rotor_arm3',
            ])

    def reset(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        super().reset(env_ids)

    def post_physics_step(self):
        self.randomize_buf += 1
        super().post_physics_step()


class ShadowHandB(CustomRandomizer, ShadowHand):
    """BayesSim-enabled version of ShadowHand."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        ShadowHand.__init__(self, cfg, sim_params, physics_engine,
                            device_type, device_id, headless)
        # Correct 0.0 -> 0.1 for properties that will be ramdomized by scaling.
        # TODO: Ask IG team to address this issue more systematically.
        dr_params = self.cfg['task']['randomization_params']
        for actor, actor_properties in dr_params['actor_params'].items():
            for env_id in list(range(self.num_envs)):
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                num_dofs = self.gym.get_actor_dof_count(env, handle)
                dof_prop = self.gym.get_actor_dof_properties(env, handle)
                for j in range(num_dofs):
                    if dof_prop['stiffness'][j] <= 0.0:
                        dof_prop['stiffness'][j] = 0.1
                        if env_id == 0:
                            print('Init 0.0->1.0 for stiffness of joint',
                                  self.gym.get_actor_dof_names(env, handle)[j])
                self.gym.set_actor_dof_properties(env, handle, dof_prop)
        # Now can initialize ParamsGenerator.
        self.actor_params_generator = ParamsGenerator(
            self.gym, self.envs[0], self.cfg['task']['randomization_params'],
            # plot: 'robot0:thdistal', 'object'
            plot_names_skip_patterns=[
                'robot0:hand mount', 'robot0:forearm', 'robot0:wrist',
                'robot0:palm',
                'robot0:ffknuckle', 'robot0:ffproximal',
                'robot0:ffmiddle', 'robot0:ffdistal', 'robot0:mfknuckle',
                'robot0:mfproximal', 'robot0:mfmiddle', 'robot0:mfdistal',
                'robot0:rfknuckle', 'robot0:rfproximal', 'robot0:rfmiddle',
                'robot0:rfdistal', 'robot0:lfmetacarpal', 'robot0:lfknuckle',
                'robot0:lfproximal', 'robot0:lfmiddle', 'robot0:lfdistal',
                'robot0:thbase', 'robot0:thproximal', 'robot0:thhub',
                'robot0:thmiddle',
                # 'robot0:thdistal',
                'robot0:WRJ1', 'robot0:WRJ0', 'robot0:FFJ3', 'robot0:FFJ2',
                'robot0:FFJ1', 'robot0:FFJ0', 'robot0:MFJ3', 'robot0:MFJ2',
                'robot0:MFJ1', 'robot0:MFJ0', 'robot0:RFJ3', 'robot0:RFJ2',
                'robot0:RFJ1', 'robot0:RFJ0', 'robot0:LFJ4', 'robot0:LFJ3',
                'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0', 'robot0:THJ4',
                'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0',
                # 'T_RFJ1c',
                'T_MFJ1c', 'T_RFJ1c', 'T_LFJ1c',
                # 'object',
            ])

    def reset(self, env_ids, goal_env_ids):
        super().reset(env_ids, goal_env_ids)
        # IG udercounts 1 step in compute_hand_reward:
        # torch.where(progress_buf >= max_episode_length, ...
        # other envs have something like:
        # reset = torch.where(progress_buf >= max_episode_length - 1,
        # TODO: Ask IG team to make termination consistent.
        self.progress_buf[env_ids] = 1


class VecTaskPythonB(VecTaskPython):
    # A general note: IG will set reset bit on the last step (i.e. the step
    # right before reset). This is in contrast to OpenAI vectorized envs that
    # usually set done bit on the 1st step of next episode (i.e. after reset).

    def __init__(self, task, device):
        # IG clips observations to [-5,5]. This could result in information loss
        # on the space boundaries, so clip obs to 100 instead.
        super().__init__(task, device, clip_observations=100.0,
                         clip_actions=1.0)

    # Note: reset() is not a part of the BaseTask API; VecTaskPython reset()
    # method simply does a step without calling task.reset().
    # For BayesSim we need to actually reset the task, hence this custom
    # reset function.
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.task.num_envs).to(self.task.device)
        freq = self.task.cfg['task']['randomization_params'].get('frequency', 1)
        self.task.randomize_buf += freq  # make sure to reset randomization
        self.task.reset_buf += 1
        if isinstance(self.task, ShadowHandB):
            self.task.reset(env_ids, env_ids)
        else:
            self.task.reset(env_ids)
        sim_params = self.task.gym.get_sim_params(self.task.sim)
        old_dt = sim_params.dt
        sim_params.dt = 1.0/5000.0  # imperceptible step
        self.task.gym.set_sim_params(self.task.sim, sim_params)
        actions = torch.zeros(self.task.num_envs, self.task.num_actions)
        self.task.progress_buf -= 1  # step will increment
        self.task.step(actions.to(self.rl_device))  # populates *_buf tensors
        obs = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs)
        sim_params.dt = old_dt
        self.task.gym.set_sim_params(self.task.sim, sim_params)
        return obs.to(self.rl_device)


def make_ig_env(args, cfg, cfg_train):
    if 'asset' in cfg['env']:
        assert(os.getenv('ISAACGYM_PATH') is not None), \
            'Please set Isaac Gym path: export ISAACGYM_PATH=/path/to/isaacgym'
        cfg['env']['asset']['assetRoot'] = os.path.join(
            os.path.expanduser(os.environ.get('ISAACGYM_PATH')), 'assets')
    ig_sim_params = parse_sim_params(args, cfg, cfg_train)
    task = eval(args.task+'B')(  # derived classes support BayesSim DR
        cfg=cfg,
        sim_params=ig_sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless)
    env = VecTaskPythonB(task, args.rl_device)
    return env
