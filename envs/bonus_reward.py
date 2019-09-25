from typing import Dict, Optional, Tuple, List, NewType

import numpy as np
import torch
import torch.distributions as ds
from distributions import FixedCategorical

class BonusReward(object):
    def __init__(self, env, beta=1.0, initial_value=1.0):
        self.beta = beta
        width = env.grid.width
        height = env.grid.height
        self.count_grid = np.zeros(
            (width, height), dtype='float32')
        # self.initial_value = np.ones(
        #     (width, height), dtype='float32') * initial_value

    # def query_bonus(self, x_array, y_array, kl_values):
    #     bonus_value = self.beta * np.multiply(self.initial_value,
    #         1 / np.sqrt(self.count_grid))
    #
    #     # self.count_grid[x_array, y_array] += 1
    #     # np.add.at(self.count_grid, (x_array, y_array), 1)
    #
    #     kl_grid = np.zeros_like(self.count_grid)
    #     t_count = np.zeros_like(self.count_grid)
    #
    #     np.add.at(t_count, (x_array, y_array), 1)
    #     np.add.at(kl_grid, (x_array, y_array), kl_values)
    #
    #     kl_grid = kl_grid / (t_count + 1e-10)
    #
    #     self.count_grid += t_count
    #     return bonus_value[x_array, y_array], \
    #         self.beta / np.sqrt(self.count_grid.T), kl_grid.T

    # def make_grid(self, x_array, y_array, kl_values, visit_count):
    #     inv_sqrt_count = 1 / np.sqrt(visit_count)
    #     bonus_value = self.beta * inv_sqrt_count * kl_values
    #
    #     kl_grid = np.zeros_like(self.count_grid)
    #     bonus_grid = np.zeros_like(self.count_grid)
    #     isq_count_grid = np.zeros_like(self.count_grid)
    #     t_count = np.zeros_like(self.count_grid)
    #
    #     np.add.at(t_count, (x_array, y_array), 1)
    #     np.add.at(kl_grid, (x_array, y_array), kl_values)
    #     np.add.at(bonus_grid, (x_array, y_array), bonus_value)
    #     np.add.at(isq_count_grid, (x_array, y_array), inv_sqrt_count)
    #
    #     kl_grid = kl_grid / (t_count + 1e-10)
    #     bonus_grid = bonus_grid / (t_count + 1e-10)
    #     isq_count_grid = isq_count_grid / (t_count + 1e-10)
    #
    #     return isq_count_grid, kl_grid, bonus_grid


def bonus_kl_forward(
    bonus_type,
    obs,
    b_args,
    bonus_z_encoder,
    masks,
    bonus_normalization,
):
    num_steps, num_processes = obs['image'].shape[:2]
    if b_args.hier_mode == 'vic':
        omega_dim_current = b_args.omega_dim_current

        omega_option = torch.eye(omega_dim_current).to(masks.device)
        if omega_dim_current < b_args.omega_option_dims:
            _diff = b_args.omega_option_dims - omega_dim_current
            _pad = omega_option.new_zeros(omega_dim_current, _diff)
            omega_option = torch.cat([omega_option, _pad], 1)

        omega_option = omega_option.unsqueeze(1).unsqueeze(1)
        omega_option = omega_option.repeat(1, num_steps, num_processes, 1)
        new_masks = masks.unsqueeze(0).repeat(omega_option.shape[0], 1, 1, 1)

        obs = obs.unsqueeze(0).repeat_dim(0, omega_option.shape[0])
        obs['omega'] = omega_option

    if bonus_type == 'kl-pi':
        bonus_policy = bonus_z_encoder
        _rhx = torch.zeros(num_steps * num_processes,
            bonus_policy.encoder_recurrent_hidden_state_size).to(masks.device)

        obs_flat = obs.flatten_three()
        new_masks_flat = new_masks.view(-1, *new_masks.shape[3:])
        z_latent, z_log_prob, z_dist, _ = \
            bonus_policy.encoder_forward(
                obs=obs_flat,
                rnn_hxs=_rhx,
                masks=new_masks_flat,
                do_z_sampling=False)

        obs_flat.pop('omega')
        obs_flat.update({'z_latent': z_latent})

        _rhx = torch.zeros(num_steps * num_processes,
            bonus_policy.recurrent_hidden_state_size).to(masks.device)

        _, action_dist, _, _ = \
            bonus_policy.get_action_dist(
                inputs=obs_flat, rnn_hxs=_rhx, masks=new_masks_flat)

        action_probs = action_dist.probs
        action_probs = action_probs.view(
            omega_option.shape[0],
            num_steps,
            num_processes,
            *action_probs.shape[1:])

        pi_kl = []
        pi_def = FixedCategorical(probs=action_probs.mean(0))
        for opt_idx in range(omega_option.shape[0]):
            pi_opt_dist = FixedCategorical(probs=action_probs[opt_idx])
            pi_kl.append(ds.kl.kl_divergence(pi_opt_dist, pi_def))

        kld = torch.stack(pi_kl, 0).unsqueeze(-1)

    else:
        if b_args.hier_mode == 'infobot-supervised':
            # _rhx = torch.zeros(num_steps * num_processes,
            obs.pop('mission')
            _rhx = torch.zeros(num_processes,
                bonus_z_encoder.recurrent_hidden_state_size).to(masks.device)

            z_dist, _, _ = bonus_z_encoder._encode(
                obs=obs.flatten_two(),
                rnn_hxs=_rhx,
                masks=masks.view(-1, *masks.shape[2:]),
            )

            z_prior = ds.normal.Normal(
                loc=torch.zeros_like(z_dist.loc),
                scale=torch.ones_like(z_dist.scale),
            )

            kld = ds.kl.kl_divergence(z_dist, z_prior)
            kld = kld.view(num_steps, num_processes, -1).sum(-1, keepdim=True)
        else:
            _rhx = torch.zeros(num_steps * num_processes,
                bonus_z_encoder.recurrent_hidden_state_size).to(masks.device)

            z_dist, _, _ = bonus_z_encoder._encode(
                obs=obs.flatten_three(),
                rnn_hxs=_rhx,
                masks=new_masks.view(-1, *new_masks.shape[3:]),
            )

            z_prior = ds.normal.Normal(
                loc=torch.zeros_like(z_dist.loc),
                scale=torch.ones_like(z_dist.scale),
            )

            kld = ds.kl.kl_divergence(z_dist, z_prior)
            kld = kld.view(omega_option.shape[0], num_steps, num_processes, -1)

    if b_args.hier_mode != 'infobot-supervised':
        if bonus_type == 'kl_max':
            kld = kld.max(0)[0].sum(-1, keepdim=True)
        else:
            # Marginalizing along omega dimention and summing across z-dimension
            kld = kld.mean(0).sum(-1, keepdim=True)

        if bonus_normalization == 'max_min':
            raise NotImplementedError
            kld = (kld - kld.min())/kld.max()
        assert bonus_normalization == 'unnormalized'

    return kld


def bonus_kl_forward_td(
    bonus_type,
    obs,
    b_args,
    bonus_z_encoder,
    masks,
    bonus_normalization,
):
    num_steps, num_processes = obs['image'].shape[:2]
    if b_args.hier_mode == 'vic':
        omega_dim_current = b_args.omega_dim_current

        omega_option = torch.eye(omega_dim_current).to(masks.device)
        if omega_dim_current < b_args.omega_option_dims:
            _diff = b_args.omega_option_dims - omega_dim_current
            _pad = omega_option.new_zeros(omega_dim_current, _diff)
            omega_option = torch.cat([omega_option, _pad], 1)

        omega_option = omega_option.unsqueeze(1).unsqueeze(1)
        omega_option = omega_option.repeat(1, num_steps, num_processes, 1)
        new_masks = masks.unsqueeze(0).repeat(omega_option.shape[0], 1, 1, 1)

        obs = obs.unsqueeze(0).repeat_dim(0, omega_option.shape[0])
        obs['omega'] = omega_option

    if bonus_type == 'kl-pi':
        bonus_policy = bonus_z_encoder
        _rhx = torch.zeros(num_steps * num_processes,
            bonus_policy.encoder_recurrent_hidden_state_size).to(masks.device)

        obs_flat = obs.flatten_three()
        new_masks_flat = new_masks.view(-1, *new_masks.shape[3:])
        z_latent, z_log_prob, z_dist, _ = \
            bonus_policy.encoder_forward(
                obs=obs_flat,
                rnn_hxs=_rhx,
                masks=new_masks_flat,
                do_z_sampling=False)

        obs_flat.pop('omega')
        obs_flat.update({'z_latent': z_latent})

        _rhx = torch.zeros(num_steps * num_processes,
            bonus_policy.recurrent_hidden_state_size).to(masks.device)

        _, action_dist, _, _ = \
            bonus_policy.get_action_dist(
                inputs=obs_flat, rnn_hxs=_rhx, masks=new_masks_flat)

        action_probs = action_dist.probs
        action_probs = action_probs.view(
            omega_option.shape[0],
            num_steps,
            num_processes,
            *action_probs.shape[1:])

        pi_kl = []
        pi_def = FixedCategorical(probs=action_probs.mean(0))
        for opt_idx in range(omega_option.shape[0]):
            pi_opt_dist = FixedCategorical(probs=action_probs[opt_idx])
            pi_kl.append(ds.kl.kl_divergence(pi_opt_dist, pi_def))

        kld = torch.stack(pi_kl, 0).unsqueeze(-1)

    else:
        if b_args.hier_mode == 'infobot-supervised':
            # _rhx = torch.zeros(num_steps * num_processes,
            obs.pop('mission')
            _rhx = torch.zeros(num_processes,
                bonus_z_encoder.recurrent_hidden_state_size).to(masks.device)

            z_dist, _, _ = bonus_z_encoder._encode(
                obs=obs.flatten_two(),
                rnn_hxs=_rhx,
                masks=masks.view(-1, *masks.shape[2:]),
            )

            z_prior = ds.normal.Normal(
                loc=torch.zeros_like(z_dist.loc),
                scale=torch.ones_like(z_dist.scale),
            )

            kld = ds.kl.kl_divergence(z_dist, z_prior)
            kld = kld.view(num_steps, num_processes, -1).sum(-1, keepdim=True)
        else:
            _rhx = torch.zeros(num_steps * num_processes,
                bonus_z_encoder.recurrent_hidden_state_size).to(masks.device)

            z_dist, _, _ = bonus_z_encoder._encode(
                obs=obs.flatten_three(),
                rnn_hxs=_rhx,
                masks=new_masks.view(-1, *new_masks.shape[3:]),
            )

            z_prior = ds.normal.Normal(
                loc=torch.zeros_like(z_dist.loc),
                scale=torch.ones_like(z_dist.scale),
            )

            kld = ds.kl.kl_divergence(z_dist, z_prior)
            kld = kld.view(omega_option.shape[0], num_steps, num_processes, -1)

    if b_args.hier_mode != 'infobot-supervised':
        if bonus_type == 'kl_max':
            kld = kld.max(0)[0].sum(-1, keepdim=True)
        else:
            # Marginalizing along omega dimention and summing across z-dimension
            kld = kld.mean(0).sum(-1, keepdim=True)

        if bonus_normalization == 'max_min':
            raise NotImplementedError
            kld = (kld - kld.min())/kld.max()
        assert bonus_normalization == 'unnormalized'

    return kld
