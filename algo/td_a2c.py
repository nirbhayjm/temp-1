# Adapted from github.com/ikostrikov/pytorch-a2c-ppo-acktr/

from typing import Dict, Optional, Tuple, Type

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds

from distributions import FixedCategorical
from envs.bonus_reward import bonus_kl_forward_td

from .kfac import KFACOptimizer


class A2C_ACKTR():
    def __init__(
        self,
        actor_critic,
        options_policy,
        # options_decoder,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        alpha=None,
        max_grad_norm=None,
        use_entropy_reg=False,
        acktr=False,
        normalize_advantage=False,
        use_max_ent=False,
        max_ent_action_logprob_coeff=None,
        model='hier',
        bonus_reward=None,
        bonus_noise_scale=0.0,
        continuous_state_space=False,
        vis_env=None,
    ):

        self.actor_critic = actor_critic
        self.options_policy = options_policy
        self.acktr = acktr
        self.continuous_state_space = continuous_state_space

        self.value_loss_coef = value_loss_coef
        self.use_entropy_reg = use_entropy_reg
        self.entropy_coef = entropy_coef

        self.lr = lr
        self.eps = eps
        self.alpha = alpha

        self.normalize_advantage = normalize_advantage
        self.use_max_ent = use_max_ent
        self.max_grad_norm = max_grad_norm
        self.max_ent_action_logprob_coeff = \
            max_ent_action_logprob_coeff

        self.model = model

        if bonus_reward is not None:
            self.bonus_reward = bonus_reward
            self.bonus_noise_scale = bonus_noise_scale
            self.vis_env = vis_env
            self.init_bonus_noise()

        self.init_optims()

    def init_optims(self):
        if self.acktr:
            raise NotImplementedError

        # if self.model == 'cond':
            # if self.options_decoder is not None:
            #     params = [
            #             {'params' : self.actor_critic.parameters()},
            #             {'params' : self.options_decoder.parameters()},
            #     ]
            # else:
        params = self.actor_critic.parameters()
        self.optimizer = optim.RMSprop(
                params, lr=self.lr, eps=self.eps, alpha=self.alpha)

        # else:
        #     # self.options_optim = optim.RMSprop(
        #     #     self.options_decoder.parameters(),
        #     #     lr=self.lr, eps=self.eps, alpha=self.alpha)
        #     # self.actor_critic_optim = optim.RMSprop(
        #     #     self.actor_critic.parameters(),
        #     #     alpha=self.alpha,
        #     #     lr=self.lr, eps=self.eps)
        #     self.optimizer = optim.Adam(
        #         self.actor_critic.parameters(),
        #         lr=self.lr, eps=self.eps)
        #
        #
        #     # if self.options_policy is not None:
        #     #     self.options_policy_optim = optim.RMSprop(
        #     #         self.options_policy.parameters(),
        #     #         alpha=self.alpha,
        #     #         lr=self.lr, eps=self.eps)

    def init_bonus_noise(self):
        if not self.continuous_state_space:
            width = self.vis_env.grid.width
            height = self.vis_env.grid.height
            self.bonus_noise_grid = np.random.randn(width, height).astype('float32') \
                * self.bonus_noise_scale

    def update(
        self,
        rollouts,
        next_value,
        option_space='continuous',
        # vic_only=False,
        hier_mode='default',
        use_intrinsic_control=False,
        # trajectory_encoder=None,
        # trajectory_optim=None,
        # traj_enc_loss_coeff=1.0,
        use_ib=False,
        ib_kwargs=None,
        # kl_optim_mode='analytic',
        # infobot_beta=None,
        agent_pos=None,
        bonus_z_encoder=None,
        b_args=None,
        bonus_type='count',
        bonus_normalization=None,
        heuristic_ds=None,
        heuristic_coeff=0.0,
        visit_count=None,
    ):

        # Validate args
        # assert hier_mode in ['default', 'vic', 'transfer', 'bonus',
        #     'infobot-supervised', 'tlvm']
        assert hier_mode in ['bonus']
        assert option_space in ['continuous', 'discrete']
        # assert kl_optim_mode in ['analytic', 'mc_sampling']
        ic_info, ib_info, option_info = {}, {}, {}

        # z_latent_dims = rollouts.z_latents.size()[-1]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _, action_dist = \
            self.actor_critic.evaluate_actions(
                inputs=rollouts.obs[:-1].flatten_two(),
                rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                masks=rollouts.masks[:-1].view(-1, 1),
                action=rollouts.actions.view(-1, action_shape),
                get_entropy=True,
            )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # Sanity check assertion
        with torch.no_grad():
            _diff = (action_log_probs - rollouts.action_log_probs) * rollouts.masks[:-1]
            _diff = (_diff ** 2).mean()
            assert _diff < 1e-8, "Action log probs mismatch with mse: {}".format(_diff)
            # if _diff > 1e-8:
            #     import pdb; pdb.set_trace()
            #     print("What?!?!?!")

        r_t_additives = rollouts.rewards.new_zeros(rollouts.rewards.shape)
        path_derivative_terms = rollouts.rewards.new_zeros(1)

        # 1 - Max ent
        if self.use_max_ent:
            r_t_additives -= action_log_probs * rollouts.masks[:-1]\
                * self.max_ent_action_logprob_coeff
            path_derivative_terms -= (action_log_probs * rollouts.masks[:-1]\
                * self.max_ent_action_logprob_coeff).mean()

        # 2 - Exploration bonus, either visitation count based or KL based
        with torch.no_grad():
            final_corrected_obs = \
                rollouts.prev_final_obs.mul(rollouts.prev_final_mask) \
                + rollouts.obs[:-1].mul(1 - rollouts.prev_final_mask)

            visit_count = torch.from_numpy(
                visit_count.astype('float32')).to(
                rollouts.masks.device).unsqueeze(-1)
            corrected_visit_count = \
                (rollouts.prev_final_visit_count * rollouts.prev_final_mask) \
                + (visit_count * (1 - rollouts.prev_final_mask))

            r_t_bonus, kl_bonus = self.kl_exploration_bonus(
                agent_pos=agent_pos,
                masks=rollouts.masks,
                obs=final_corrected_obs,
                bonus_z_encoder=bonus_z_encoder,
                bonus_type=bonus_type,
                bonus_normalization=bonus_normalization,
                visit_count=corrected_visit_count,
                rollouts=rollouts,
                heuristic_ds=heuristic_ds,
                heuristic_coeff=heuristic_coeff,
                b_args=b_args,
                ic_info=ic_info,
            )
            if bonus_type != 'count':
                r_t_bonus *= kl_bonus
            ic_info['bonus_tensor'] = r_t_bonus.cpu().numpy()
            ic_info['bonus_reward'] = r_t_bonus.mean().item()
        r_t_additives += r_t_bonus

        # Return computation as sum of discounted (rewards + additives)
        returns = rollouts.compute_returns(
            rewards=rollouts.rewards,
            masks=rollouts.masks,
            value_preds=rollouts.value_preds,
            next_value=next_value,
            step_additives=r_t_additives,
        )[:-1]

        value_target = returns.detach()
        advantages = returns - values.detach()

        if self.normalize_advantage:
            _mean = advantages.mean(1, keepdim=True)
            _var = ((advantages - _mean) ** 2).mean(1, keepdim=True)
            _std = torch.sqrt(_var)
            advantages = (advantages - _mean) / (_std + 1e-5)

        # option_advantages = option_returns - values
        ic_info['effective_return'] = returns.mean().item()

        # Action loss computation
        action_loss = -(advantages.detach() * action_log_probs)
        action_loss = (action_loss * rollouts.masks[:-1]).mean()

        value_loss = F.mse_loss(
            values * rollouts.masks[:-1],
            value_target * rollouts.masks[:-1],
            reduction='elementwise_mean') * self.value_loss_coef

        actor_critic_loss = action_loss + value_loss
        # path_derivative_loss = -1 * returns.mean()
        path_derivative_loss = -1 * path_derivative_terms.mean()

        if self.use_entropy_reg:
            actor_critic_loss -= dist_entropy * self.entropy_coef

        self.optimizer.zero_grad()
        (actor_critic_loss + path_derivative_loss).backward()
        self.clip_grad_all([self.actor_critic])
        self.optimizer.step()

        with torch.no_grad():
            ic_info.update(ib_info)
            action_dist_probs = action_dist.probs.view(
                num_steps, num_processes, action_dist.probs.shape[1])
            ic_info['action_dist_probs'] = action_dist_probs.cpu().numpy()
            action_log_probs_mean = (action_log_probs * rollouts.masks[:-1]).mean()

            # Sanity check assertion
            assert action_log_probs_mean.item() > -10

        return value_loss.item(), action_loss.item(), \
            dist_entropy.item(), action_log_probs_mean.item(), \
            ic_info, option_info

    def update_infobot_supervised(
        self,
        rollouts,
        next_value,
        infobot_beta,
        anneal_coeff,
    ):
        # Validate args
        # assert kl_optim_mode in ['analytic', 'mc_sampling']
        ic_info = {}

        # z_latent_dims = rollouts.z_latents.size()[-1]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        z_sample, z_gauss_dist, values, action_log_probs, \
        dist_entropy, _, action_dist = \
            self.actor_critic.evaluate_actions(
                inputs=rollouts.obs[:-1].flatten_two(),
                rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                masks=rollouts.masks[:-1].view(-1, 1),
                action=rollouts.actions.view(-1, action_shape),
                z_eps=rollouts.z_eps.view(-1, self.actor_critic.z_latent_size),
                get_entropy=True,
            )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # Sanity check assertion
        with torch.no_grad():
            _diff = (action_log_probs - rollouts.action_log_probs) * rollouts.masks[:-1]
            _diff = (_diff ** 2).mean()
            assert _diff < 1e-8, "Action log probs mismatch with mse: {}".format(_diff)
            # if _diff > 1e-8:
            #     import pdb; pdb.set_trace()
            #     print("What?!?!?!")

        r_t_additives = rollouts.rewards.new_zeros(rollouts.rewards.shape)
        path_derivative_terms = rollouts.rewards.new_zeros(1)

        # 1 - Max ent
        if self.use_max_ent:
            r_t_additives -= action_log_probs * rollouts.masks[:-1]\
                * self.max_ent_action_logprob_coeff
            path_derivative_terms -= (action_log_probs * rollouts.masks[:-1]\
                * self.max_ent_action_logprob_coeff).mean()

        # Infobot bottleneck
        z_prior = ds.normal.Normal(
            loc=torch.zeros_like(z_sample),
            scale=torch.ones_like(z_sample),
        )
        kld_zz = ds.kl.kl_divergence(z_gauss_dist, z_prior)
        kld_zz = kld_zz.view(num_steps, num_processes,
            self.actor_critic.z_latent_size)
        kld_zz = kld_zz.sum(-1, keepdim=True)

        r_t_additives -= kld_zz * infobot_beta * anneal_coeff
        path_derivative_terms -= kld_zz.mean() * infobot_beta * anneal_coeff

        # Return computation as sum of discounted (rewards + additives)
        returns = rollouts.compute_returns(
            rewards=rollouts.rewards,
            masks=rollouts.masks,
            value_preds=rollouts.value_preds,
            next_value=next_value,
            step_additives=r_t_additives,
        )[:-1]

        value_target = returns.detach()
        advantages = returns - values.detach()

        if self.normalize_advantage:
            _mean = advantages.mean(1, keepdim=True)
            _var = ((advantages - _mean) ** 2).mean(1, keepdim=True)
            _std = torch.sqrt(_var)
            advantages = (advantages - _mean) / (_std + 1e-5)

        with torch.no_grad():
            # option_advantages = option_returns - values
            ic_info['effective_return'] = returns.mean().item()
            ic_info['zz_kld'] = kld_zz.mean().item()
            ic_info['zz_kl_loss'] = \
                kld_zz.mean().item() * infobot_beta * anneal_coeff

        # Action loss computation
        action_loss = -(advantages.detach() * action_log_probs)
        action_loss = (action_loss * rollouts.masks[:-1]).mean()

        value_loss = F.mse_loss(
            values * rollouts.masks[:-1],
            value_target * rollouts.masks[:-1],
            reduction='elementwise_mean') * self.value_loss_coef

        actor_critic_loss = action_loss + value_loss
        # path_derivative_loss = -1 * returns.mean()
        path_derivative_loss = -1 * path_derivative_terms.mean()

        if self.use_entropy_reg:
            actor_critic_loss -= dist_entropy * self.entropy_coef

        self.optimizer.zero_grad()
        (actor_critic_loss + path_derivative_loss).backward()
        self.clip_grad_all([self.actor_critic])
        self.optimizer.step()

        with torch.no_grad():
            action_dist_probs = action_dist.probs.view(
                num_steps, num_processes, action_dist.probs.shape[1])
            ic_info['action_dist_probs'] = action_dist_probs.cpu().numpy()
            action_log_probs_mean = (action_log_probs * rollouts.masks[:-1]).mean()

            # Sanity check assertion
            assert action_log_probs_mean.item() > -10

        return value_loss.item(), action_loss.item(), \
            dist_entropy.item(), action_log_probs_mean.item(), \
            ic_info


    def clip_grad_all(self, nets):
        """Clip gradient by norm for all networks in list of nets"""
        for net in nets:
            if net is not None:
                nn.utils.clip_grad_norm_(
                    net.parameters(), self.max_grad_norm)

    def kl_exploration_bonus(
        self,
        agent_pos,
        obs,
        masks,
        bonus_z_encoder,
        bonus_type,
        bonus_normalization,
        visit_count,
        rollouts,
        heuristic_ds,
        heuristic_coeff,
        b_args,
        ic_info,
    ):
        # Part 1: Z-KL values
        if bonus_type != 'count':
            # # Part 1: Z-KL values
            kld = bonus_kl_forward_td(
                obs=obs,
                b_args=b_args,
                bonus_z_encoder=bonus_z_encoder,
                bonus_type=bonus_type,
                masks=masks[1:],
                bonus_normalization=bonus_normalization,
            )
        else:
            kld = torch.zeros_like(masks[1:])

        # # Ignore Z-KL for Z sampeld after last time step
        # kld = kld[:-1]

        # Part 2: Visitation count
        r_t_bonus = torch.zeros_like(masks[1:])
        agent_pos = torch.from_numpy(agent_pos).to(masks.device)
        # plus_masks = torch.eq(
        #     torch.cat([(1 + (0 * masks[0:1])), masks[:-1]], 0), 1)
        plus_masks = masks[1:]

        visitation_x = agent_pos[1:, :, 0:1].masked_select(
            torch.eq(plus_masks, 1)).cpu().numpy()
        visitation_y = agent_pos[1:, :, 1:2].masked_select(
            torch.eq(plus_masks, 1)).cpu().numpy()
        # kl_values = kld.masked_select(
        #     torch.eq(plus_masks, 1)).cpu().numpy()

        # isq_count_grid, kl_grid, bonus_grid = self.bonus_reward.make_grid(
        #     x_array=visitation_x,
        #     y_array=visitation_y,
        #     kl_values=kl_values,
        #     visit_count=visit_count.reshape(-1),
        # )
        # ic_info['bonus_kl_grid'] = kl_grid
        # ic_info['bonus_isq_grid'] = isq_count_grid
        # ic_info['bonus_grid'] = bonus_grid

        # flat_bonus = torch.from_numpy(flat_bonus).to(masks.device)
        # r_t_bonus.masked_scatter_(plus_masks, flat_bonus)

        # Skipping first state visitation bonus
        # r_t_bonus = r_t_bonus[1:]

        # visit_count = torch.from_numpy(
        #     visit_count.astype('float32')).to(masks.device)
        # visit_count = visit_count[1:]
        inv_sqrt_count = 1 / torch.sqrt(visit_count)

        # Noise added to visit count
        bonus_noise = self.bonus_noise_grid[visitation_x, visitation_y]
        bonus_noise = torch.from_numpy(bonus_noise).to(masks.device)

        heuristic_ds = torch.from_numpy(heuristic_ds).float().to(masks.device)
        coeff = (heuristic_ds * heuristic_coeff) \
            + ((1-heuristic_ds) * self.bonus_reward.beta)
        # count_bonus = self.bonus_reward.beta * inv_sqrt_count
        count_bonus = coeff.unsqueeze(-1) * inv_sqrt_count
        bonus_noise_tensor = torch.zeros_like(count_bonus)
        bonus_noise_tensor.masked_scatter_(plus_masks.byte(), bonus_noise)
        count_bonus += bonus_noise_tensor

        r_t_bonus += count_bonus
        return r_t_bonus, kld
