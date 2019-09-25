import time
import numpy as np
import scipy
import os, sys
import json
from collections import namedtuple

import torch
import torch.distributions as ds

import gym
from gym.envs.registration import register
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from distributions import FixedCategorical
import utilities.utilities as utils
from storage import DictObs
from envs.bonus_reward import bonus_kl_forward

XYTuple = namedtuple("XYTuple", ["x", "y"])

def dict_stack_helper(list_of_dicts):
    return {key:np.stack([item[key] for item in list_of_dicts], 0) \
        for key, val in list_of_dicts[0].items() if type(val) != str}

def eval_ib_kl(args, vis_env, actor_critic, device, omega_dim_current,
    num_samples=10):
    assert args.use_infobot != 0
    action_dims = vis_env.action_space.n
    if hasattr(vis_env.actions, 'forward'):
        action_space_type = 'pov'
    elif hasattr(vis_env.actions, 'up'):
        action_space_type = 'cardinal'

    vis_obs, vis_info = vis_env.reset()
    assert 'rgb_grid' in vis_info
    env_rgb_img = vis_info['rgb_grid'].transpose([2, 0, 1])
    env_rgb_img = np.flip(env_rgb_img, 1)

    all_obs = vis_env.enumerate_states()

    _rhx = torch.zeros(num_samples,
        actor_critic.recurrent_hidden_state_size).to(device)
    _masks = torch.ones(1, num_samples, 1).to(device)

    def repeat_dict_obs(dict_obs, batch_size):
        out = {}
        for key, value in dict_obs.items():
            out[key] = np.broadcast_to(value[np.newaxis, :], (batch_size, *value.shape))
        return out

    grid_shape = (vis_env.width, vis_env.height)
    # kl_zz_grid = torch.zeros(*grid_shape).to(device)
    kl_zz_opt_grid = [torch.zeros(*grid_shape).to(device) \
        for _ in range(omega_dim_current)]
    kl_pi_def_grid = torch.zeros(*grid_shape).to(device)
    kl_pi_opt_grid = [torch.zeros(*grid_shape).to(device) \
        for _ in range(omega_dim_current)]
    pi_def_grid = torch.zeros((action_dims, *grid_shape)).to(device)
    pi_opt_grid = [torch.zeros((action_dims, *grid_shape)).to(device) \
        for _ in range(omega_dim_current)]

    if args.option_space == 'continuous':
        _shape = (num_samples, args.omega_option_dims)
        _loc = torch.zeros(*_shape).to(device)
        _scale = torch.ones(*_shape).to(device)
        omega_prior = ds.normal.Normal(loc=_loc, scale=_scale)

        _z_shape = (num_samples, args.z_latent_dims)
        _z_loc = torch.zeros(*_z_shape).to(device)
        _z_scale = torch.ones(*_z_shape).to(device)
        z_prior = ds.normal.Normal(loc=_z_loc, scale=_z_scale)

        for key, obs in all_obs.items():
            obs = repeat_dict_obs(obs, num_samples)
            omega_option = omega_prior.rsample()

            obs = DictObs({key:torch.from_numpy(obs_i).to(device) \
                for key, obs_i in obs.items()})

            if 'mission' in obs.keys():
                obs.pop('mission')
            obs.update({'omega': omega_option})

            z_latent, z_log_prob, z_dist, _ = \
                actor_critic.encoder_forward(
                    obs=obs,
                    rnn_hxs=_rhx,
                    masks=_masks,
                    do_z_sampling=True)

            kld_zz = ds.kl.kl_divergence(z_dist, z_prior)
            # kld_zz = kld_zz.view(
            #     num_steps + 1, num_processes, z_latent_dims)
            kld_zz = torch.sum(kld_zz, 1).mean()

            kl_zz_grid[key.x, key.y] = kld_zz

    else:
        # _shape = (omega_dim_current, args.omega_option_dims)
        # uniform_probs = torch.ones(*_shape).to(device)

        _z_shape = (omega_dim_current * num_samples, args.z_latent_dims)
        _z_loc = torch.zeros(*_z_shape).to(device)
        _z_scale = torch.ones(*_z_shape).to(device)
        z_prior = ds.normal.Normal(loc=_z_loc, scale=_z_scale)

        # if omega_dim_current < args.omega_option_dims:
        #     uniform_probs[:, omega_dim_current:].fill_(0)
        # uniform_probs = uniform_probs / uniform_probs.sum(-1, keepdim=True)
        # omega_prior = FixedCategorical(probs=uniform_probs)
        # # option_discrete = omega_prior.sample()

        omega_option = torch.eye(omega_dim_current).to(device)
        if omega_dim_current < args.omega_option_dims:
            _diff = args.omega_option_dims - omega_dim_current
            _pad = omega_option.new_zeros(omega_dim_current, _diff)
            omega_option = torch.cat([omega_option, _pad], 1)
        omega_option = omega_option.unsqueeze(0).repeat(num_samples, 1, 1)
        omega_option = omega_option.view(-1, *omega_option.shape[2:])

        for key, obs in all_obs.items():
            obs = repeat_dict_obs(obs, omega_option.shape[0])
            # omega_option = omega_prior.rsample()

            obs = DictObs({key:torch.from_numpy(obs_i).to(device) \
                for key, obs_i in obs.items()})

            if 'mission' in obs.keys():
                obs.pop('mission')
            obs.update({'omega': omega_option})

            z_latent, z_log_prob, z_dist, _ = \
                actor_critic.encoder_forward(
                    obs=obs,
                    rnn_hxs=_rhx,
                    masks=_masks,
                    do_z_sampling=True)

            kld_zz = ds.kl.kl_divergence(z_dist, z_prior)
            kld_zz = kld_zz.view(
                num_samples, omega_dim_current, *kld_zz.shape[1:])
            kld_zz = kld_zz.sum(-1).mean(0)
            for opt_idx in range(omega_dim_current):
                kl_zz_opt_grid[opt_idx][key.x, key.y] = kld_zz[opt_idx]

            obs.pop('omega')
            obs.update({'z_latent': z_latent})

            _, action_dist, _, _ = \
                actor_critic.get_action_dist(
                    inputs=obs, rnn_hxs=_rhx, masks=_masks)

            action_probs = action_dist.probs
            action_probs = action_probs.view(
                num_samples, omega_dim_current, *action_probs.shape[1:]).mean(0)

            pi_opt, pi_kl = {}, {}
            for opt_idx in range(omega_dim_current):
                pi_opt[opt_idx] = FixedCategorical(probs=action_probs[opt_idx])
                pi_opt_grid[opt_idx][:, key.x, key.y] = pi_opt[opt_idx].probs
            pi_def = FixedCategorical(probs=action_probs.mean(0))
            pi_def_grid[:, key.x, key.y] = pi_def.probs

            for opt_idx in range(omega_dim_current):
                pi_kl[opt_idx] = ds.kl.kl_divergence(pi_opt[opt_idx], pi_def)
                kl_pi_opt_grid[opt_idx][key.x, key.y] = pi_kl[opt_idx]

            pi_kl_avg = torch.stack(tuple(pi_kl.values()), 0).mean(0)
            kl_pi_def_grid[key.x, key.y] = pi_kl_avg

    pi_opt_grid = torch.stack(pi_opt_grid, 0)
    kl_pi_opt_grid = torch.stack(kl_pi_opt_grid, 0)
    kl_zz_opt_grid = torch.stack(kl_zz_opt_grid, 0)
    kl_zz_grid = kl_zz_opt_grid.mean(0)

    return_dict = {
        'env_rgb_img': env_rgb_img,
        'pi_opt_grid': pi_opt_grid.cpu().numpy().transpose([0, 1, 3, 2]),
        'pi_def_grid': pi_def_grid.cpu().numpy().transpose([0, 2, 1]),
        'kl_zz_grid': kl_zz_grid.cpu().numpy().T,
        'kl_zz_opt_grid': kl_zz_opt_grid.cpu().numpy().transpose([0, 2, 1]),
        'kl_pi_def_grid': kl_pi_def_grid.cpu().numpy().T,
        'kl_pi_opt_grid': kl_pi_opt_grid.cpu().numpy().transpose([0, 2, 1]),
    }
    return return_dict

def eval_success(
    args,
    val_envs,
    vis_env,
    actor_critic,
    b_args,
    bonus_type,
    bonus_z_encoder,
    bonus_beta,
    bonus_normalization,
    device,
    num_episodes,
):
    ARGMAX_POLICY = True
    episode_count = 0
    return_list = []
    all_max_room = []
    val_envs.modify_attr('render_rgb', [True] * args.num_processes)
    val_envs.reset_config_rng()
    # vis_env.reset_config_rng()
    grid_shape = (vis_env.width, vis_env.height)

    kl_grid = torch.zeros(*grid_shape).to(device)
    bonus_grid = torch.zeros(*grid_shape).to(device)

    while episode_count < num_episodes:
        reward_list = []
        reset_output = val_envs.reset()

        obs = reset_output[:, 0]
        info = reset_output[:, 1]
        obs = dict_stack_helper(obs)
        info = dict_stack_helper(info)
        obs = DictObs({key:torch.from_numpy(obs_i).to(device) \
        for key, obs_i in obs.items()})

        rgb_grid = info['rgb_grid']

        recurrent_hidden_states = torch.zeros(args.num_processes,
        actor_critic.recurrent_hidden_state_size).to(device)
        masks = torch.ones(args.num_processes, 1).to(device)
        agent_pos = [val_envs.get_attr('agent_pos')]
        all_masks = [np.array([True]*args.num_processes)]
        all_obs = [obs]
        all_vc = [np.ones(args.num_processes)]

        for step in range(args.num_steps):
            _, action, _, recurrent_hidden_states = \
                actor_critic.act(
                    inputs=obs,
                    rnn_hxs=recurrent_hidden_states,
                    masks=masks,
                    deterministic=bool(ARGMAX_POLICY))

            cpu_actions = action.view(-1).cpu().numpy()

            obs, reward, _, info = val_envs.step(cpu_actions)
            reward_list.append(reward)

            obs = dict_stack_helper(obs)
            obs = DictObs({key:torch.from_numpy(obs_i).to(device) \
                for key, obs_i in obs.items()})
            all_obs.append(obs)

            done = np.stack([item['done'] for item in info], 0)
            curr_pos = np.stack([item['agent_pos'] for item in info], 0)
            # curr_dir = np.stack([item['agent_dir'] for item in info], 0)
            visit_count = np.stack([item['visit_count'] for item in info], 0)
            all_vc.append(visit_count)
            agent_pos.append(curr_pos)
            all_masks.append(done == False)

            if 'max_room_id' in info[0]:
                max_room = np.stack([item['max_room_id'] for item in info], 0)
            else:
                max_room = np.ones((args.num_processes)) * -1

        agent_pos = np.stack(agent_pos, 0)
        all_masks = np.stack(all_masks, 0)
        all_max_room.append(max_room)

        stacked_obs = {}
        for key in all_obs[0].keys():
            stacked_obs[key] = torch.stack(
                [_obs[key] for _obs in all_obs], 0)
        stacked_obs = DictObs(stacked_obs)
        stacked_masks = np.stack(all_masks, 0).astype('float32')
        stacked_masks = torch.from_numpy(stacked_masks).to(device)

        stacked_visit_count = np.stack(all_vc, 0)

        if bonus_type != 'count':
            bonus_kld = bonus_kl_forward(
                bonus_type=bonus_type,
                obs=stacked_obs,
                b_args=b_args,
                bonus_z_encoder=bonus_z_encoder,
                masks=stacked_masks,
                bonus_normalization=bonus_normalization,
            )
        else:
            bonus_kld = stacked_masks.clone() * 0

        episodic_return = np.stack(reward_list, 0).sum(0)
        return_list.append(episodic_return)
        episode_count += args.num_processes

    VIS_COUNT = 1
    VIS_IDX = 0
    agent_pos = agent_pos[:, VIS_IDX]
    episode_length = all_masks[:, VIS_IDX].sum()
    rgb_env_image = rgb_grid[VIS_IDX]
    bonus_kld = bonus_kld[:, VIS_IDX]
    visit_count = stacked_visit_count[:, VIS_IDX]
    rgb_env_image = np.flip(rgb_env_image.transpose([2, 0, 1]), 1)

    vis_info = make_bonus_grid(
        bonus_beta=bonus_beta,
        agent_pos=agent_pos,
        kl_values=bonus_kld.squeeze(-1).cpu().numpy(),
        visit_count=visit_count,
        episode_length=episode_length,
        grid_shape=grid_shape,
    )
    vis_info['rgb_env_image'] = rgb_env_image

    all_return = np.concatenate(return_list, 0)
    success = (all_return > 0).astype('float')
    all_max_room = np.concatenate(all_max_room, 0)

    return success, all_max_room, vis_info

def eval_success_simple(
    num_processes,
    num_steps,
    val_envs,
    actor_critic,
    device,
    num_episodes,
):
    ARGMAX_POLICY = True
    episode_count = 0
    return_list = []
    all_max_room = []
    val_envs.modify_attr('render_rgb', [False] * num_processes)
    val_envs.reset_config_rng()

    while episode_count < num_episodes:
        reward_list = []
        reset_output = val_envs.reset()

        obs = reset_output[:, 0]
        info = reset_output[:, 1]
        obs = dict_stack_helper(obs)
        info = dict_stack_helper(info)
        obs = DictObs({key:torch.from_numpy(obs_i).to(device) \
        for key, obs_i in obs.items()})

        recurrent_hidden_states = torch.zeros(num_processes,
        actor_critic.recurrent_hidden_state_size).to(device)
        masks = torch.ones(num_processes, 1).to(device)

        for step in range(num_steps):
            _, action, _, recurrent_hidden_states = \
                actor_critic.act(
                    inputs=obs,
                    rnn_hxs=recurrent_hidden_states,
                    masks=masks,
                    deterministic=bool(ARGMAX_POLICY))

            cpu_actions = action.view(-1).cpu().numpy()

            obs, reward, _, info = val_envs.step(cpu_actions)
            reward_list.append(reward)

            obs = dict_stack_helper(obs)
            obs = DictObs({key:torch.from_numpy(obs_i).to(device) \
                for key, obs_i in obs.items()})

            done = np.stack([item['done'] for item in info], 0)

            if 'max_room_id' in info[0]:
                max_room = np.stack([item['max_room_id'] for item in info], 0)
            else:
                max_room = np.ones((num_processes)) * -1

        all_max_room.append(max_room)

        episodic_return = np.stack(reward_list, 0).sum(0)
        return_list.append(episodic_return)
        episode_count += num_processes

    all_return = np.concatenate(return_list, 0)
    success = (all_return > 0).astype('float')
    all_max_room = np.concatenate(all_max_room, 0)

    return success, all_max_room, all_return

def make_bonus_grid(
    bonus_beta,
    agent_pos,
    kl_values,
    visit_count,
    episode_length,
    grid_shape,
):
    x_array = agent_pos[:, 0][:episode_length]
    y_array = agent_pos[:, 1][:episode_length]
    visit_count = visit_count[:episode_length]
    kl_values = kl_values[:episode_length]

    inv_sqrt_count = 1 / np.sqrt(visit_count)
    bonus_value = bonus_beta * inv_sqrt_count * kl_values

    kl_grid = np.zeros(grid_shape)
    bonus_grid = np.zeros(grid_shape)
    isq_count_grid = np.zeros(grid_shape)
    t_count = np.zeros(grid_shape)

    np.add.at(t_count, (x_array, y_array), 1)
    np.add.at(kl_grid, (x_array, y_array), kl_values)
    np.add.at(bonus_grid, (x_array, y_array), bonus_value)
    np.add.at(isq_count_grid, (x_array, y_array), inv_sqrt_count)

    kl_grid_avg = kl_grid / (t_count + 1e-10)
    bonus_grid_avg = bonus_grid / (t_count + 1e-10)
    isq_count_grid_avg = isq_count_grid / (t_count + 1e-10)

    return {
        'isq_count_grid': isq_count_grid.T,
        'kl_grid': kl_grid.T,
        'bonus_grid': bonus_grid.T,
        'isq_count_grid_avg': isq_count_grid_avg.T,
        'kl_grid_avg': kl_grid_avg.T,
        'bonus_grid_avg': bonus_grid_avg.T,
        't_count': t_count.T,
    }
