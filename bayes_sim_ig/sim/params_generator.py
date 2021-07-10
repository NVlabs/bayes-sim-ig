# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""An interface for generating simulation params as a flat numpy array."""

import numpy as np

from isaacgym.gymutil import get_property_getter_map

PARAM_NAMES_LINK_PREFIXES = ('rigid_body_properties', 'rigid_shape_properties')
PARAM_NAMES_JOINT_PREFIXES = ('dof_properties', 'tendon_properties')


def get_names(gym, env, handle):
    body_names = gym.get_actor_rigid_body_names(env, handle)
    body_shape_ids = gym.get_actor_rigid_body_shape_indices(env, handle)
    dof_names = gym.get_actor_dof_names(env, handle)
    shape_names = []
    for body_id, shape_ids_info in enumerate(body_shape_ids):
        assert(shape_ids_info.start == len(shape_names))
        if shape_ids_info.count == 0:
            continue  # nothing to do
        shape_names.append(body_names[body_id])
        if shape_ids_info.count > 1:
            for tmp_i in range(1,shape_ids_info.count):
                shape_names.append(body_names[body_id]+'_'+str(tmp_i))
    tendon_names = []
    num_tendons = gym.get_actor_tendon_count(env, handle)
    for i in range(num_tendons):
        tendon_names.append(gym.get_actor_tendon_name(env, handle, i))
    return body_names, shape_names, dof_names, tendon_names


def make_name(body_names, shape_names, dof_names, tendon_names,
              name_skip_patterns, oper, prop_name, prop_idx, attr_name,
              attr_idx=None):
    sfx = '_'+attr_name
    if attr_idx is not None:
        sfx += '_'+str(attr_idx)
    if prop_name == 'rigid_body_properties':
        name = body_names[prop_idx]+sfx
    elif prop_name == 'rigid_shape_properties':
        name = shape_names[prop_idx]+sfx
    elif prop_name == 'tendon_properties':
        name = tendon_names[prop_idx]+sfx
    elif ((prop_name == 'dof_properties') and
          (attr_idx is not None) and (prop_idx == 0)):
        name = dof_names[attr_idx]+'_'+attr_name
    else:
        name = prop_name+'_'+str(prop_idx)+sfx
    if oper == 'scaling':
        name += '_mult'
    skip = False
    if name_skip_patterns is not None:
        for pattern in name_skip_patterns:
            if pattern in name:
                skip = True
    return name, skip


def check_operation(operation, default, name, prop_name, attr_name):
    if operation == 'scaling':
        assert(default > 0), f'Error: operation scaling zero default {name:s}'
        # lo_hi = lo_hi * default
    elif operation == 'additive':
        assert(default == 0), f'Error: operation additive needs default==0' \
                              f' for {name:s} {prop_name:s} {attr_name:s} ' \
                              f' but got {default:0.4f}'
        # lo_hi = lo_hi + default
    else:
        assert(False), f'Unknown operation{operation:s}'


class ParamsGenerator:
    def __init__(self, gym, env, dr_params, plot_names_skip_patterns=None,
                 body_names=None, shape_names=None, dof_names=None,
                 tendon_names=None):
        self._distr = None  # set by BayesSim main code
        self._defaults, self._names, self._lows, self._highs, \
        self._skip_ids = self.get_actor_params_info(
                gym, env, dr_params, plot_names_skip_patterns,
                body_names, shape_names, dof_names, tendon_names)
        print('Created ParamsGenerator with', len(self._lows), 'dims:')
        for nm, dflt, low, high in zip(self.names, self.defaults,
                                       self.lows, self.highs):
            print(f'{nm:s} range [{low:0.6f} {high:0.6f}] default {dflt:0.6f}')

    @property
    def names(self):
        return self._names

    @property
    def lows(self):
        return self._lows

    @property
    def highs(self):
        return self._highs

    @property
    def defaults(self):
        return self._defaults

    @property
    def skip_ids(self):
        return self._skip_ids

    def set_distr(self, distr):
        self._distr = distr

    def sample(self):
        flat_smpl = self._distr.gen(n_samples=1)[0]
        flat_smpl = np.clip(flat_smpl, self._lows, self._highs)
        return flat_smpl

    def get_actor_params_info(self, gym, env, dr_params,
                              plot_names_skip_patterns=None,
                              body_names=None, shape_names=None, dof_names=None,
                              tendon_names=None):
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(gym)
        curr_id = 0
        skip_ids = []
        for actor_name, actor_properties in dr_params['actor_params'].items():
            handle = None
            if env is not None:
                handle = gym.find_actor_handle(env, actor_name)
                body_names, shape_names, dof_names, tendon_names = get_names(
                    gym, env, handle)
            else:
                assert(body_names is not None), 'Please specify body_names'
                assert(shape_names is not None), 'Please specify shape_names'
                assert(dof_names is not None), 'Please specify dof_names'
                assert(tendon_names is not None), 'Please specify tendon_names'
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                if prop_name == 'scale':  # assuming apply_randomizations.py
                    lo_hi = np.array(prop_attrs['range'])
                    oper = prop_attrs['operation']
                    param = gym.get_actor_scale(env, handle)
                    check_operation(oper, param, actor_name, 'scale', '')
                    name = actor_name+'_scale'
                    if oper == 'scaling':
                        name += '_mult'
                    if plot_names_skip_patterns is not None:
                        for pattern in plot_names_skip_patterns:
                            if pattern in name:
                                skip_ids.append(curr_id)
                    params.append(param)
                    names.append(name)
                    lows.append(lo_hi[0])
                    highs.append(lo_hi[1])
                    continue
                props = [None]
                if handle is not None:
                    props = param_getters_map[prop_name](env, handle)
                    if not isinstance(props, list):
                        props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr_name, attr_randomization_params in prop_attrs.items():
                        lo_hi = np.array(attr_randomization_params['range'])
                        oper = attr_randomization_params['operation']
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr_name].shape[0]):
                                param = prop[attr_name][attr_idx]
                                name, skip = make_name(
                                    body_names, shape_names, dof_names,
                                    tendon_names, plot_names_skip_patterns,
                                    oper, prop_name, prop_idx, attr_name,
                                    attr_idx)
                                if skip:
                                    skip_ids.append(curr_id)
                                check_operation(oper, param, name,
                                                prop_name, attr_name)
                                params.append(param)
                                names.append(name)
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                                curr_id += 1
                        else:
                            param = 1.0
                            if prop is not None:
                                param = getattr(prop, attr_name)
                            name, skip = make_name(
                                body_names, shape_names, dof_names,
                                tendon_names, plot_names_skip_patterns, oper,
                                prop_name, prop_idx, attr_name)
                            if skip:
                                skip_ids.append(curr_id)
                            check_operation(oper, param, name,
                                            prop_name, attr_name)
                            params.append(param)
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
                            curr_id += 1
        return np.array(params), np.array(names), \
               np.array(lows), np.array(highs), skip_ids
