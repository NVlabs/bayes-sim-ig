# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""A vectorized implementation of basic OpenAI envs (e.g. Pendulum)"""

import os

from gym import spaces
from gym.utils import seeding
import numpy as np
import torch

from rlgpu.tasks.base.base_task import BaseTask

from isaacgym import gymapi

from .params_generator import ParamsGenerator


class PendulumB(BaseTask):
    """Vectorized OpenAI Pendulum with IG API."""
    def __init__(self, cfg, sim_params, physics_engine,
                 device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.device_type = device_type
        self.graphics_device_id = device_id
        self.headless = headless
        self.simple_viewer = None
        self.cfg['env']['numObservations'] = 3
        self.cfg['env']['numActions'] = 1
        self.max_episode_length = cfg['env']['episodeLength']
        BaseTask.__init__(self, cfg)
        self.actor_params_generator = ParamsGenerator(
            self.gym, None, self.cfg['task']['randomization_params'],
            body_names=['pendulum'], shape_names=['pendulum'],
            dof_names=['pendulum'], tendon_names=[])
        self.length_dim = self.mass_dim = None
        for idx, nm in enumerate(self.actor_params_generator.names):
            if 'length' in nm:
                self.length_dim = idx
            if 'mass' in nm:
                self.mass_dim = idx
        assert(self.length_dim is not None)
        assert(self.mass_dim is not None)
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self._lengths = np.ones((self.num_envs, ))
        self._masses = np.ones((self.num_envs, ))
        self._states = np.zeros((self.num_envs, 2))
        self._last_us = np.ones((self.num_envs, 1))
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque,
            shape=(1,), dtype=np.float32)
        self.action_rng = torch.from_numpy(
            self.action_space.high - self.action_space.low)

    def denorm_action(self, action):
        return action*self.max_torque  # assuming action in [-1,1]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)

    def reset(self, env_ids):
        env_ids = env_ids.detach().cpu().numpy()
        for env_id in env_ids:
            smpl = self.actor_params_generator.sample()
            self.extern_actor_params[env_id] = smpl
            self._lengths[env_id] = smpl[self.length_dim]
            self._masses[env_id] = smpl[self.mass_dim]
            high = np.array([np.pi, 1])
            # Assuming np randomness has been initialized with a seed.
            rnd = np.random.uniform(low=-high, high=high,
                                    size=(env_ids.shape[0], 2))
            self._states[env_ids, :] = rnd[:]
        self.obs_buf[:] = torch.tensor(self._get_obs()).float().to(self.device)
        self.rew_buf[:] = 0.0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # Apply actions (convert from [-1,1]), get rewards; reset when needed.
        acts = (actions*self.max_torque).detach().cpu().numpy()
        self._last_us[:] = acts
        self.progress_buf[:] += 1  # has to be updated before reset
        if (self.reset_buf > 0).any():
            # Reset has to happen before obs,rwd computation in IG.
            assert((self.reset_buf > 0).all())  # fixed length episodes
            self.reset(self.reset_buf.nonzero(as_tuple=False).squeeze(-1))
            self.obs_buf[:] = torch.tensor(
                self._get_obs()).float().to(self.device)
            self.rew_buf[:] = torch.tensor(
                self._compute_reward(acts[:, 0])).float().to(self.device)
        else:
            # Take a step in the env, then update obs,rwd,progress,reset bufs.
            obs, rwd, _, _ = self._step(acts)
            self.obs_buf[:] = torch.tensor(obs).float().to(self.device)
            self.rew_buf[:] = torch.tensor(rwd).float().to(self.device)
        self.reset_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf), self.reset_buf)
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def get_img(self, env_id=0, height=200, width=200):
        if self.simple_viewer is None:  # init
            from gym.envs.classic_control import rendering
            self.simple_viewer = rendering.Viewer(height, width)
            self.simple_viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.simple_viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.simple_viewer.add_geom(axle)
            fname = os.path.join(os.path.dirname(__file__),
                                 'assets', 'clockwise.png')
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
        self.simple_viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self._states[env_id][0] + np.pi/2)
        if self._last_us[env_id]:
            self.imgtrans.scale = (-self._last_us[env_id] / 2,
                                   np.abs(self._last_us[env_id]) / 2)
        return self.simple_viewer.render(return_rgb_array=True)

    def override_state(self, obs):
        theta_cos, theta_sin, thetadot = torch.unbind(obs, dim=-1)
        theta = torch.atan2(theta_sin, theta_cos)
        self._states[:, 0] = theta
        self._states[:, 1] = thetadot

    @staticmethod
    def angle_normalize(x):
        return ((x+np.pi) % (2*np.pi)) - np.pi

    def _get_obs(self):
        theta = self._states[:, 0]
        thetadot = self._states[:, 1]
        obs = np.column_stack([np.cos(theta), np.sin(theta), thetadot])
        return obs

    def _step(self, u):
        u = np.clip(u[:, 0], -self.max_torque, self.max_torque)
        th = self._states[:, 0]
        thdot = self._states[:, 1]
        g = 10.
        dt = self.dt
        rwd = self._compute_reward(u)
        newthdot = thdot + (-3*g/(2*self._lengths) * np.sin(th + np.pi) +
                            3./(self._masses*self._lengths**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        self._states = np.column_stack([newth, newthdot])
        return self._get_obs(), rwd, False, {}

    def _compute_reward(self, u):
        th = self._states[:, 0]
        thdot = self._states[:, 1]
        costs = PendulumB.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        return -costs
