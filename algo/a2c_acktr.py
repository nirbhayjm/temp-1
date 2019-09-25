# Adapted from github.com/ikostrikov/pytorch-a2c-ppo-acktr/

from typing import Dict, Optional, Tuple, Type

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds

from distributions import FixedCategorical
from envs.bonus_reward import bonus_kl_forward

from .kfac import KFACOptimizer


class A2C_ACKTR():
    def __init__(
        self,
        actor_critic,
        options_policy,
        options_decoder,
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
        """Actor-Critic Algorithm Class

        Used to train (upto) the following 4 networks:
            - actor_critic: The parametrized policy i.e. pi(a | s, *)
            - trajectory_encoder: Inference network required for any intrinsic
                control objective (VIC, VALOR, etc.) i.e. q(Omega | Tau)
                Note that the trajectory_encoder and trajectory_optim are
                not given as arguments and are instead passed to the update
                method. This creates a logical separation from "policy" and
                "inference" network types.
            - options_policy: Higher level policy pi(Omega | s) used for
                hierarchical control along with a lower-level actor_critic
                policy pi(a | s, Omega)
            - options_decoder: Legacy network, currently unused.

        Arguments:
        actor_critic: pi(a | s, *) network
        options_policy: pi(Omega | s) network
        options_decoder: legacy argument, unused

        value_loss_coef: Coefficient for critic's value loss
        use_entropy_reg: Boolean flag for using entropy regularization
        entropy_coef: Coefficient for entropy regularization
        use_max_ent: Boolean flag for using maximum-entropy objective
        max_ent_action_logprob_coeff: Coefficient for maximum-entropy objective

        normalize_advantage: Boolean flag for using batch normalization on
            the advantage
        max_grad_norm: Gradient clip-by-norm upper bound
        alpha, eps, lr: Optimizer arguments
        acktr: Whether to use A2C with Kronecker-Factored Trust Region (ACKTR),
            this was supported for a while but has now been deactivated and
            will raise a NotImplementedError.
        model: Legacy argument, always set this to 'hier' for now.

        """

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
        self.options_decoder = None

        if options_decoder is not None:
            self.options_decoder = options_decoder

        if bonus_reward is not None:
            self.bonus_reward = bonus_reward
            self.bonus_noise_scale = bonus_noise_scale
            self.vis_env = vis_env
            self.init_bonus_noise()

        self.init_optims()

    def init_optims(self):
        if self.acktr:
            assert self.model == 'hier'
            self.options_optim = optim.RMSprop(
                self.options_decoder.parameters(),
                lr=self.lr, eps=self.eps, alpha=self.alpha)
            self.actor_critic_optim = KFACOptimizer(self.actor_critic)
        else:
            if self.model == 'cond':
                if self.options_decoder is not None:
                    params = [
                            {'params' : self.actor_critic.parameters()},
                            {'params' : self.options_decoder.parameters()},
                    ]
                else:
                    params = self.actor_critic.parameters()
                self.optimizer = optim.RMSprop(
                        params, lr=self.lr, eps=self.eps, alpha=self.alpha)

            else:
                # self.options_optim = optim.RMSprop(
                #     self.options_decoder.parameters(),
                #     lr=self.lr, eps=self.eps, alpha=self.alpha)
                self.actor_critic_optim = optim.RMSprop(
                    self.actor_critic.parameters(),
                    alpha=self.alpha,
                    lr=self.lr, eps=self.eps)
                if self.options_policy is not None:
                    self.options_policy_optim = optim.RMSprop(
                        self.options_policy.parameters(),
                        alpha=self.alpha,
                        lr=self.lr, eps=self.eps)

    def init_bonus_noise(self):
        if not self.continuous_state_space:
            width = self.vis_env.grid.width
            height = self.vis_env.grid.height
            self.bonus_noise_grid = np.random.randn(width, height).astype('float32') \
                * self.bonus_noise_scale

    def update(
        self,
        rollouts,
        option_space='continuous',
        # vic_only=False,
        hier_mode='default',
        use_intrinsic_control=False,
        ic_kwargs=None,
        trajectory_encoder=None,
        trajectory_optim=None,
        traj_enc_loss_coeff=1.0,
        # infobot_mode=False,
        use_ib=False,
        ib_kwargs=None,
        # kl_optim_mode='analytic',
        # option_log_probs=None,
        # infobot_beta=None,
        # z_prior=None,
        agent_pos=None,
        bonus_z_encoder=None,
        b_args=None,
        bonus_type='count',
        bonus_normalization=None,
        visit_count=None,
        heuristic_ds=None,
        heuristic_coeff=None,
    ):
        """A2C Update of actor critic model and other networks

        rollouts:
            An instance of RolloutStorage which holds trajectories used
            for the A2C update and live variables (still part of
            computational graph) for the intrinsic control and IB objectives.

        option_space:
            'discrete' or 'continuous' corresponding to the option space.

        hier_mode:
            'default': Vanilla policy update of `actor_critic` network pi(A | S),
                currently unsupported and will raise a NotImplementedError

            'vic': Intrinsic control update mode, updates `actor_critic` policy
                network pi(A | S, Omega) and `trajectory_encoder` inference
                network q(Omega | Tau) using an intrinsic control objective,
                handled by the `intrinsic_control_loss` method. This method has
                it's argument passed via `ic_kwargs`.

            'transfer': Hierarchical policy update of `actor_critic` network
                pi(A | S, Omega) and a higher level option selection network
                `options_policy` pi(Omega | S) which may operate at different
                time horizons.

        Information bottlenecked policy update:
            `use_ib` flag and `ib_kwargs` are given to the `infobot_loss`
            method for computing the terms in the IB objective. Forward pass
            if modified to first compute Z from S, Omega and then using a
            policy pi(A | S, Z).


        """

        # Validate args
        assert hier_mode in ['default', 'vic', 'transfer', 'bonus', 'infobot-supervised']
        assert option_space in ['continuous', 'discrete']
        # assert kl_optim_mode in ['analytic', 'mc_sampling']
        ic_info, ib_info, option_info = {}, {}, {}

        # z_latent_dims = rollouts.z_latents.size()[-1]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        if self.model == 'cond':
            pass

        elif hier_mode == 'infobot-supervised':
            pass

        elif hier_mode == 'transfer':
            """Hierarchical transfer: Forward pass of pi(Omega | S)"""
            dense_option_samples = []
            opt_masks = []
            opt_rewards = []
            num_option_steps = len(rollouts.option_intervals)
            rollouts.option_intervals.append(num_steps)
            opt_interval_tensor = rollouts.rewards.new_tensor(
                data=rollouts.option_intervals, dtype=torch.long)

            # Loop over time intervals where a higher level option was sampled
            for idx in range(len(rollouts.option_intervals[:-1])):
                _interval_start = rollouts.option_intervals[idx]
                _interval_len = rollouts.option_intervals[idx + 1] \
                    - rollouts.option_intervals[idx]
                _interval_end = _interval_start + _interval_len
                dense_option_samples.append(
                    rollouts.omega_option_steps[idx].unsqueeze(0).repeat(_interval_len, 1, 1))
                opt_masks.append(rollouts.masks[rollouts.option_intervals[idx]])
                opt_rewards.append(rollouts.rewards[_interval_start:_interval_end].sum(0))

            option_obs = rollouts.obs[opt_interval_tensor[:-1]]
            omega_option = torch.cat(dense_option_samples, 0)
            omega_option_dims = omega_option.shape[-1]
            omega_option_original = torch.stack(rollouts.omega_option_steps, 0)
            omega_option_original = omega_option_original.view(-1, omega_option_dims)
            if option_space == 'discrete':
                # assert option_log_probs is not None
                omega_option_original = torch.argmax(omega_option_original, dim=1, keepdim=True)
            opt_masks = torch.stack(opt_masks, 0)
            opt_rewards = torch.stack(opt_rewards, 0)
            options_rhx = torch.stack(rollouts.options_rhx, 0)
            # option_log_probs = torch.stack(rollouts.option_log_probs, 0)
            # option_values = torch.stack(rollouts.option_values, 0)
            omega_option_flat = omega_option.view(-1, omega_option_dims)

            option_values, option_log_probs, option_entropy, _ = self.options_policy.evaluate_actions(
                # obs=rollouts.obs[0],
                # omega_option=omega_option,
                inputs=option_obs.flatten_two(),
                rnn_hxs=options_rhx[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                masks=opt_masks.view(-1, 1),
                action=omega_option_original,
                get_entropy=True)

            option_log_probs = option_log_probs.view(num_option_steps, num_processes, -1)
            option_values = option_values.view(num_option_steps, num_processes, -1)

        else:
            """Get options which were sampled from prior"""
            _, omega_option_dims = rollouts.omega_option.size()
            omega_option = rollouts.omega_option
            omega_option_flat = omega_option.unsqueeze(0).repeat(
                num_steps, 1, 1).view(-1, omega_option_dims)

            if option_space == 'discrete':
                # assert option_log_probs is not None
                omega_option = torch.argmax(omega_option, dim=1, keepdim=True)

            """
            Evaluate options: Note that options_decoder is no longer used
            but this part of the code is still kept around as options_decoder
            is a dummy network for the purposes of intrinsic control.
            """
            option_log_probs, option_entropy = self.options_decoder.evaluate_options(
                obs=rollouts.obs[0], omega_option=omega_option)

        obs_base = rollouts.obs[:-1].flatten_two()
        if self.model != 'cond' and hier_mode != 'infobot-supervised':
            # Replacing the stale omega in rollouts.obs (which is not part of
            # the computational graph, as it was used with torch.no_grad())
            # with the live omega which _is_ a part of the computational graph
            obs_base.pop('mission')
            obs_base.update({'omega': omega_option_flat})

        if use_intrinsic_control:
            """Compute intrinsic control loss in `ic_kl_objective`"""
            ic_kl_objective, ic_batch_weights, ic_adaptive_wts, ic_info = \
                self.intrinsic_control_loss(**ic_kwargs,
                    rollouts=rollouts, omega_option=omega_option, hier_mode=hier_mode)
        else:
            ic_info.update({
                'empowerment_value': 0,
                'p_ll': 0,
                'batch_weights': np.ones((1)),
                'kld_qp': 0,
                'q_ll': 0,
                'p_entropy': 0,
                'q_entropy': 0,
                'pq_loss': 0,
                'log_det_j': 0,
            })

        if use_ib:
            """Bottlenecked policy evaluation from forward pass

                - Uses the latent Z variables computed by p(Z | S, Omega) in
                  forward pass and substitutes the option Omega with latent Z
                  in `obs_base` so that it may be subsequently be used by
                  the policy pi(A | S, Z).

                - Computes the IB objective terms in `ib_kl_objective`
            """
            obs_base, ib_kl_objective, default_action_dist, ib_info = \
                self.infobot_loss(
                    **ib_kwargs,
                    rollouts=rollouts,
                    obs_base=obs_base,
                    ic_adaptive_wts=None)

        """Policy pi(A | S, *) evaluation from forward pass

        Note that `*` = Z in case of IB policy, `*` = Omega for plain
        intrinsic control, and `*` = nothing for vanilla conditional policy.
        However, only the first two are supported right now and a vanilla
        conditional policy is never used.
        """
        values, action_log_probs, dist_entropy, _, action_dist = \
            self.actor_critic.evaluate_actions(
                inputs=obs_base,
                rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                masks=rollouts.masks[:-1].view(-1, 1),
                action=rollouts.actions.view(-1, action_shape),
                get_entropy=True)

        # if use_ib:
        #     """Bookkeeping for IB policy"""
        with torch.no_grad():
            action_probs = action_dist.probs.view(
                num_steps, num_processes, *action_dist.probs.shape[1:])
            ib_info['action_probs'] = action_probs

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

        if hier_mode == 'vic':
            # Do not use environment reward
            # advantages *= 0.0
            rollouts.rewards *= 0.0

        """Computing returns from rewards and IC + IB + Max-Entropy objectives

        Return is computed as as the sum of discounted future rewards, where
        any extra objectives may be either added to the rewards (before the
        discounted sum computation) or to the returns (after the discounted
        sum computation), using the variables:

            `r_t_additives`: Stores objectives which are added to the reward
                per time step, before return is computed.

            `r_episodic_additives`: Stored objectives which are episodic and
                are added with equal weight to all time steps to the return.

        Additionally, the `path_derivative_terms` variable stores any path
        derivative dependendices of the IC/IB/Max-Entropy objective.
        """
        r_t_additives = rollouts.rewards.new_zeros(rollouts.rewards.shape)
        r_episodic_additives = rollouts.rewards.new_zeros(rollouts.returns.shape)
        if hier_mode == 'transfer':
            opt_r_t_additives = rollouts.rewards.new_zeros(
                num_option_steps, *rollouts.rewards.shape[1:])
            opt_r_episodic_additives = rollouts.rewards.new_zeros(
                num_option_steps + 1, *rollouts.returns.shape[1:])
        path_derivative_terms = rollouts.rewards.new_zeros(1)

        # 1 - Max ent
        if self.use_max_ent:
            r_t_additives -= action_log_probs * rollouts.masks[:-1]\
                * self.max_ent_action_logprob_coeff
            path_derivative_terms -= (action_log_probs * rollouts.masks[:-1]\
                * self.max_ent_action_logprob_coeff).mean()

            if hier_mode == 'transfer':
                opt_r_t_additives -= option_log_probs * opt_masks\
                    * self.max_ent_action_logprob_coeff
                path_derivative_terms -= (option_log_probs * opt_masks\
                    * self.max_ent_action_logprob_coeff).mean()

        # # Return computation as sum of discounted (rewards + additives)
        # returns = rollouts.compute_returns(
        #     r_t_additives)[:-1]
        # # returns = rollouts.returns[:-1]
        # # option_returns = (rollouts.rewards + r_t_additives).sum(0)

        # 2 - Intrinsic control objective
        if use_intrinsic_control:
            # ic_kl_objective, ic_batch_weights, ic_p_ll, ic_info = \
            #     self.intrinsic_control_loss(**ic_kwargs,
            #         rollouts=rollouts, omega_option=omega_option)
            if ic_kwargs['ic_mode'] in ['vic', 'valor']:
                r_episodic_additives -= ic_batch_weights * ic_kl_objective
                path_derivative_terms -= traj_enc_loss_coeff \
                    * (ic_batch_weights * ic_kl_objective).mean()
            elif ic_kwargs['ic_mode'] == 'diyan':
                ic_batch_weights = ic_batch_weights[:-1]
                r_t_additives -= ic_batch_weights * ic_kl_objective
                path_derivative_terms -= traj_enc_loss_coeff \
                    * (ic_batch_weights * ic_kl_objective).mean()
            # returns -= ic_kl_objective
            # option_returns -= ic_kl_objective.sum(0)

        # 3 - Infobot objective
        if use_ib:
            # returns -= ib_kl_objective
            # Old
            # r_episodic_additives -= ib_kl_objective #[:-1]
            # New
            r_episodic_additives[:-1] -= ib_kl_objective
            path_derivative_terms -= ib_kl_objective.mean()
            # option_returns -= ib_kl_objective.sum(0)

        # 4 - Exploration bonus, either visitation count based or KL based
        if hier_mode == 'bonus' and not self.continuous_state_space:
            with torch.no_grad():
                r_t_bonus, kl_bonus = self.kl_exploration_bonus(
                    agent_pos=agent_pos,
                    rollouts=rollouts,
                    bonus_z_encoder=bonus_z_encoder,
                    bonus_type=bonus_type,
                    bonus_normalization=bonus_normalization,
                    visit_count=visit_count,
                    heuristic_ds=heuristic_ds,
                    heuristic_coeff=heuristic_coeff,
                    b_args=b_args,
                    ic_info=ic_info,
                )
                if bonus_type != 'count':
                    r_t_bonus *= kl_bonus
                ic_info['bonus_tensor'] = r_t_bonus.cpu().numpy()
                ic_info['bonus_reward'] = r_t_bonus.mean().item()
            r_t_additives += r_t_bonus[:-1]

        # Return computation as sum of discounted (rewards + additives)
        returns = rollouts.compute_returns(
            rewards=rollouts.rewards,
            masks=rollouts.masks,
            value_preds=rollouts.value_preds,
            next_value=rollouts.next_value,
            step_additives=r_t_additives,
            episodic_additives=r_episodic_additives)[:-1]

        option_returns_mean = 0
        if hier_mode == 'transfer':
            option_returns = rollouts.compute_returns(
                rewards=opt_rewards,
                masks=torch.cat([opt_masks, rollouts.masks[-1:]], 0),
                value_preds=torch.stack(rollouts.option_values \
                    + [rollouts.next_option_value], 0),
                next_value=rollouts.next_option_value,
                step_additives=opt_r_t_additives,
                episodic_additives=opt_r_episodic_additives)[:-1]
            option_returns_mean += option_returns.mean().item()

        # returns = rollouts.returns[:-1]
        # option_returns = (rollouts.rewards + r_t_additives).sum(0)

        """Computing advantage and value target from returns"""
        value_target = returns.detach()
        advantages = returns - values.detach()

        if hier_mode == 'transfer':
            option_value_target = option_returns.detach()
            option_advantages = option_returns - option_values.detach()

        if self.normalize_advantage:
            _mean = advantages.mean(1, keepdim=True)
            _var = ((advantages - _mean) ** 2).mean(1, keepdim=True)
            _std = torch.sqrt(_var)
            advantages = (advantages - _mean) / (_std + 1e-5)
            if hier_mode == 'transfer':
                _mean = option_advantages.mean(1, keepdim=True)
                _var = ((option_advantages - _mean) ** 2).mean(1, keepdim=True)
                _std = torch.sqrt(_var)
                option_advantages = (option_advantages - _mean) / (_std + 1e-5)

        # option_advantages = option_returns - values
        ic_info['effective_return'] = returns.mean().item()

        """Action loss, value loss, option loss"""
        # Action loss computation
        # action_loss = -(advantages.detach() * action_log_probs).mean()
        action_loss = -(advantages.detach() * action_log_probs)
        action_loss = (action_loss * rollouts.masks[:-1]).mean()

        option_loss = 0
        # if option_space == 'discrete':
        if hier_mode == 'transfer':
            option_loss += -(option_advantages.detach() * option_log_probs)
            option_loss = (option_loss * opt_masks).mean()

        elif self.model == 'hier' and hier_mode != 'infobot-supervised':
            # NOTE: Zero-th advantage should be sum of non-discounted rewards
            option_loss += -(advantages[0:1].detach() \
                * option_log_probs).mean()

        # Value loss computation
        # value_loss = ((value_target - values) * rollouts.masks[:-1]).\
        #     pow(2).mean() * self.value_loss_coef
        value_loss = F.mse_loss(
            values * rollouts.masks[:-1],
            value_target * rollouts.masks[:-1],
            reduction='elementwise_mean') * self.value_loss_coef

        option_value_loss = 0
        if hier_mode == 'transfer':
            option_value_loss += F.mse_loss(
                option_values * opt_masks,
                option_value_target * opt_masks,
                reduction='elementwise_mean') * self.value_loss_coef

        with torch.no_grad():
            option_info['option_returns'] = option_returns_mean
            option_info['option_value_loss'] = option_value_loss
            option_info['option_loss'] = option_loss
            # ic_info['option_loss'] = option_loss

        """actor_critic loss and path_derivative_loss"""
        actor_critic_loss = action_loss + option_loss \
            + value_loss + option_value_loss
        # path_derivative_loss = -1 * returns.mean()
        path_derivative_loss = -1 * path_derivative_terms.mean()
        # path_derivative_loss *= 0.0
        # ic_info['path_derivative_loss'] = path_derivative_loss.item()

        # if self.use_entropy_reg and not vic_only:
        if self.use_entropy_reg:
            actor_critic_loss -= dist_entropy * self.entropy_coef
            actor_critic_loss -= option_entropy * self.entropy_coef

        # if use_intrinsic_control and self.use_max_ent:
        #     actor_critic_loss -= q_entropy * self.entropy_coef

        if self.acktr and self.actor_critic_optim.steps % self.actor_critic_optim.Ts == 0:
            """ACKTR update, currently not supported"""
            raise NotImplementedError
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.actor_critic_optim.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.actor_critic_optim.acc_stats = False

        # if use_intrinsic_control:
        if self.model != 'cond':
            if hier_mode == 'infobot-supervised':
                self.actor_critic_optim.zero_grad()
                (actor_critic_loss + path_derivative_loss).backward()
                if self.acktr == False:
                    self.clip_grad_all([
                        self.options_policy,
                        self.options_decoder,
                        self.actor_critic,
                        trajectory_encoder
                    ])
                self.actor_critic_optim.step()

            else:
                """IC (+ IB, optionally) policy optimizer step"""
                self.actor_critic_optim.zero_grad()
                trajectory_optim.zero_grad()
                if hier_mode == 'transfer':
                    self.options_policy_optim.zero_grad()

                (actor_critic_loss + path_derivative_loss).backward()
                if self.acktr == False:
                    self.clip_grad_all([
                        self.options_policy,
                        self.options_decoder,
                        self.actor_critic,
                        trajectory_encoder
                    ])
                if hier_mode == 'transfer':
                    self.options_policy_optim.step()
                else:
                    self.actor_critic_optim.step()
                    trajectory_optim.step()
                # if ic_kwargs['q_start_flag']:
                #     self.options_optim.step()

        else:
            """Vanilla conditional policy optimizer step"""
            self.optimizer.zero_grad()
            (actor_critic_loss + path_derivative_loss).backward()
            self.clip_grad_all([self.actor_critic])
            self.optimizer.step()

        with torch.no_grad():
            """Bookkeeping"""
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

    def clip_grad_all(self, nets):
        """Clip gradient by norm for all networks in list of nets"""
        for net in nets:
            if net is not None:
                nn.utils.clip_grad_norm_(
                    net.parameters(), self.max_grad_norm)

    def intrinsic_control_loss(
        self,
        ic_mode: str,
        rollouts,
        q_dist,
        omega_option,
        trajectory_encoder,
        traj_encoder_input,
        option_space,
        log_det_j,
        q_start_flag: bool,
        kl_coeff: float,
        anneal_coeff: float,
        reweight_by_omega_ll: bool,
        hier_mode: str,
        kl_optim_mode: str = 'mc_sampling',
    ):
        assert option_space in ['continuous', 'discrete']
        return_dict = {}

        # num_steps, num_processes, _ = rollouts.rewards.size()

        # Encoder trajectory and predict omega p_dist
        if ic_mode == 'vic':
            final_step_obs = rollouts.obs.gather_timesteps(
                indices=rollouts.masks.sum(0).long() - 1)

            # if 'image' in final_step_obs.keys():
            #     assert final_step_obs['image'].sum().item() > 0

            if traj_encoder_input == 'final_and_initial_state':
                trajectory = rollouts.obs[0], final_step_obs
            else:
                trajectory = final_step_obs

            p_dist = trajectory_encoder(trajectory)

        elif ic_mode == 'valor':
            p_dist = trajectory_encoder(
                trajectory=rollouts.obs,
                masks=rollouts.masks)

        elif ic_mode == 'diyan':
            for key in rollouts.obs.keys():
                any_key = key
                break
            dim0, dim1 = rollouts.obs[any_key].shape[:2]
            flat_obs = rollouts.obs.flatten_two()

            p_dist = trajectory_encoder(flat_obs, resizing_shape=(dim0, dim1))
            omega_option = omega_option.unsqueeze(0)

        else:
            raise ValueError

        if hier_mode == 'transfer':
            if option_space == 'continuous':
                p_log_prob = p_dist.log_prob(omega_option)
            else:
                p_log_prob = p_dist.log_probs(omega_option)

            with torch.no_grad():
                ic_adaptive_wts = torch.ones_like(
                    p_log_prob.sum(-1, keepdim=True))
                r_adaptive_wts = torch.ones_like(
                    p_log_prob.sum(-1, keepdim=True))
                batch_weights = torch.ones_like(p_log_prob.sum(-1, keepdim=True))
                batch_weights = batch_weights[:1]

                return_dict['kld_qp'] = 0
                return_dict['empowerment_value'] = 0
                return_dict['p_ll'] = torch.sum(p_log_prob, 1).mean().item()
                return_dict['q_ll'] = 0
                return_dict['p_entropy'] = p_dist.entropy().mean().item()
                return_dict['q_entropy'] = 0
                return_dict['batch_weights'] = batch_weights.cpu().numpy()
                return_dict['ic_adaptive_wts'] = ic_adaptive_wts
                return_dict['r_adaptive_wts'] = r_adaptive_wts
                return_dict['log_det_j'] = 0
                return_dict['pq_loss'] = 0

            p_minus_q_eff = 0
            if kl_optim_mode == 'mc_sampling':
                return -1.0 * p_minus_q_eff, batch_weights, ic_adaptive_wts, return_dict

            elif kl_optim_mode == 'analytic':
                raise NotImplementedError("q_dist detach not implemented!")

        else:
            if option_space == 'continuous':
                p_log_prob = p_dist.log_prob(omega_option)
                q_log_prob = q_dist.log_prob(omega_option)
            else:
                # NOTE: p & q are FixedCategorical dists
                p_log_prob = p_dist.log_probs(omega_option)
                q_log_prob = q_dist.log_probs(omega_option)

            if not q_start_flag:
                q_log_prob = q_log_prob.detach()

            p_minus_q = (p_log_prob - q_log_prob).sum(-1) # Sum along omega_dim

            with torch.no_grad():
                p_probs = torch.exp(p_log_prob.sum(-1))
                if option_space == 'continuous':
                    _threshold = 0.1
                    _r_threshold = 0.2
                    ic_adaptive_wts = p_minus_q / (1e-8 + torch.abs(
                        (p_dist.log_prob(p_dist.loc) - q_log_prob).sum(-1)))
                    # _normalizer = torch.abs((p_dist.log_prob(p_dist.loc) \
                    #     - q_log_prob).sum(-1))
                    # ic_adaptive_wts = torch.ge(p_minus_q,
                    #     _threshold * _normalizer)
                    r_prob_wts = ic_adaptive_wts - _r_threshold
                    ic_adaptive_wts -= _threshold
                    _masks = torch.ge(ic_adaptive_wts, 0)
                    _r_masks = torch.ge(r_prob_wts, 0)
                    ic_adaptive_wts = torch.abs(ic_adaptive_wts * _masks.float())
                    r_prob_wts = torch.abs(r_prob_wts * _r_masks.float())
                    return_dict['ic_adaptive_wts'] = ic_adaptive_wts.cpu().numpy()
                    return_dict['r_adaptive_wts'] = r_prob_wts.cpu().numpy()
                    ic_adaptive_wts = ic_adaptive_wts.unsqueeze(0).unsqueeze(-1)
                else:
                    _threshold = 0.51
                    # _r_threshold = 0.7
                    _wts = p_probs - _threshold
                    _masks = torch.ge(_wts, 0)
                    ic_adaptive_wts = torch.abs(_wts * _masks.float())
                    r_prob_wts = torch.abs(_wts * _masks.float())
                    return_dict['ic_adaptive_wts'] = ic_adaptive_wts.cpu().numpy()
                    return_dict['r_adaptive_wts'] = r_prob_wts.cpu().numpy()
                    ic_adaptive_wts = ic_adaptive_wts.unsqueeze(0).unsqueeze(-1)

                if reweight_by_omega_ll:
                    batch_weights = (1.0 - p_probs).unsqueeze(-1)
                    # batch_weights = 1.02 - p_probs
                    # batch_weights /= batch_weights.mean(0, keepdim=True)
                else:
                    batch_weights = torch.ones_like(p_log_prob.sum(-1, keepdim=True))


            # Adding log_det_j to p_minus_q
            # i.e. subtracting log_det_j from q - p
            p_minus_q = p_minus_q + log_det_j
            if ic_mode == 'diyan':
                p_minus_q = p_minus_q.unsqueeze(-1)
                # Remove first time step, as DIYAN should not try to infer
                # option/skill from initial state itself
                p_minus_q = p_minus_q[1:] * rollouts.masks[:-1]
                # NOTE: DIYAN does not give episodic rewards, do not sum across T!
                # p_minus_q = p_minus_q.sum(0, keepdim=True) # Sum along time steps
            else:
                p_minus_q = p_minus_q.unsqueeze(0).unsqueeze(-1)
            p_minus_q_eff = kl_coeff * anneal_coeff * p_minus_q

            with torch.no_grad():
                # For logging purposes only
                kld_qp_nograd = ds.kl.kl_divergence(q_dist, p_dist)
                if ic_mode == 'diyan':
                    if len(kld_qp_nograd.shape) == 2:
                        kld_qp_nograd = kld_qp_nograd.unsqueeze(-1)
                    kld_qp_nograd *= rollouts.masks
                    kld_qp_nograd = kld_qp_nograd.sum(0).squeeze(-1)
                if len(kld_qp_nograd.shape) > 1:
                    kld_qp_nograd = torch.sum(kld_qp_nograd, 1)
                kld_qp_nograd = kld_qp_nograd.mean()

                q_entropy = q_dist.entropy().mean()

                return_dict['kld_qp'] = kld_qp_nograd.item()
                if ic_mode == 'diyan':
                    return_dict['empowerment_sum_t'] = p_minus_q.sum(0).mean().item()
                    return_dict['empowerment_value'] = p_minus_q.max(0)[0].mean().item()
                    return_dict['p_ll'] = p_log_prob.max(0)[0].mean().item()
                    return_dict['q_ll'] = q_log_prob.mean().item()
                else:
                    return_dict['empowerment_value'] = p_minus_q.sum(0).mean().item()
                    return_dict['p_ll'] = torch.sum(p_log_prob, 1).mean().item()
                    return_dict['q_ll'] = torch.sum(q_log_prob, 1).mean().item()
                return_dict['p_entropy'] = p_dist.entropy().mean().item()
                return_dict['q_entropy'] = q_dist.entropy().mean().item()
                return_dict['batch_weights'] = batch_weights.cpu().numpy()

                pq_loss = -1 * p_minus_q_eff.mean()
                return_dict['pq_loss'] = pq_loss
                if isinstance(log_det_j, torch.Tensor):
                    return_dict['log_det_j'] = log_det_j.mean().item()
                else:
                    return_dict['log_det_j'] = log_det_j

                return_dict['pq_loss'] = pq_loss.item()

            if kl_optim_mode == 'mc_sampling':
                return -1.0 * p_minus_q_eff, batch_weights, ic_adaptive_wts, return_dict

            elif kl_optim_mode == 'analytic':
                raise NotImplementedError("q_dist detach not implemented!")
                kld_qp = ds.kl.kl_divergence(q_dist, p_dist)
                if len(kld_qp.shape) > 1:
                    kld_qp = torch.sum(kld_qp, 1)
                # '''DEBUG'''
                # std_normal_dist = ds.normal.Normal(
                #     loc=q_dist.loc.new_zeros(q_dist.loc.shape),
                #     scale=q_dist.scale.new_ones(q_dist.scale.shape)
                # )
                # kld_qp = ds.kl.kl_divergence(q_dist, std_normal_dist)
                # '''DEBUG'''
                kld_qp = kld_qp - log_det_j
                kld_qp = kld_qp.unsqueeze(0).unsqueeze(-1)
                kld_qp_eff = kl_coeff * anneal_coeff * kld_qp
                return kld_qp_eff, batch_weights, ic_adaptive_wts, return_dict

    def infobot_loss(
        self,
        rollouts,
        z_dist_type: str,
        infobot_beta: float,
        infobot_kl_coeff: float,
        kl_optim_mode: str,
        obs_base,
        ic_adaptive_wts,
        min_ib_coeff: float,
        ib_adaptive: bool,
    ):
        assert z_dist_type == 'gaussian'
        assert kl_optim_mode in ['mc_sampling', 'analytic']
        return_dict = {}

        z_latents = torch.stack(rollouts.z_latents, 0) #[:-1]
        ib_rnn_hxs = torch.stack(rollouts.ib_enc_hidden_states, 0) #[:-1]

        if isinstance(rollouts.z_dists[0], ds.normal.Normal):
            locs = [dist.loc for dist in rollouts.z_dists]
            stds = [dist.scale for dist in rollouts.z_dists]
            loc = torch.stack(locs, 0) #[:-1]
            std = torch.stack(stds, 0) #[:-1]
            loc = loc.view(-1, *loc.shape[2:])
            std = std.view(-1, *std.shape[2:])
            z_dist = ds.normal.Normal(loc=loc, scale=std)

        elif isinstance(rollouts.z_dists[0], ds.categorical.Categorical):
            raise NotImplementedError
        else:
            raise ValueError("Cannot identify z_dist type: {}"\
                .format(type(rollouts.z_dists[0])))

        z_latent_dims = z_latents.size()[-1]
        return_dict['z_latent_dims'] = z_latent_dims
        num_steps, num_processes, _ = rollouts.rewards.size()

        obs_omega = obs_base.clone()
        # z_latents, z_dist, z_logprobs, ib_rnn_hxs = \
        #     self.actor_critic.evaluate_z_latents(
        #         obs=obs_omega,
        #         rnn_hxs=ib_rnn_hxs.view(-1, *ib_rnn_hxs.shape[2:]),
        #         masks=rollouts.masks[:-1].view(-1, 1),
        #         z_latent=z_latents.view(-1, z_latent_dims),
        #     )
        if 'omega' in obs_base:
            obs_base.pop('omega')
        if 'goal_vector' in obs_base:
            obs_base.pop('goal_vector')

        obs_base.update({
            'z_latent': z_latents[:-1].view(-1, z_latent_dims),
        })
        z_latents = z_latents.view(-1, z_latent_dims)
        z_logprobs = z_dist.log_prob(z_latents)

        z_prior = ds.normal.Normal(
            loc=torch.zeros_like(z_latents),
            scale=torch.ones_like(z_latents),
        )

        if kl_optim_mode == 'mc_sampling':
            raise NotImplementedError
            z_q_ll = z_logprobs
            z_p_ll = z_prior.log_prob(z_latents)
            z_q_minus_p = z_q_ll - z_p_ll
            z_q_minus_p = z_q_minus_p.view(
                num_steps + 1, num_processes, z_latent_dims)
            kld_zz = torch.sum(z_q_minus_p, 2, keepdim=True)
        else:
            kld_zz = ds.kl.kl_divergence(z_dist, z_prior)
            kld_zz = kld_zz.view(
                num_steps + 1, num_processes, z_latent_dims)
            kld_zz = kld_zz[:-1] # Removing last time step
            kld_zz = torch.sum(kld_zz, 2, keepdim=True)

        with torch.no_grad():
            kld_zz_nograd = ds.kl.kl_divergence(z_dist, z_prior)
            kld_zz_nograd = kld_zz_nograd.view(
                num_steps + 1, num_processes, z_latent_dims)
            kld_zz_nograd = torch.sum(kld_zz_nograd, 2, keepdim=True)

            z_q_ll_nograd = z_logprobs
            z_p_ll_nograd = z_prior.log_prob(z_latents)
            q_minus_p_nograd = z_q_ll_nograd - z_p_ll_nograd
            q_minus_p_nograd = q_minus_p_nograd.view(
                num_steps + 1, num_processes, z_latent_dims)
            q_minus_p_nograd = torch.sum(q_minus_p_nograd, 2, keepdim=True)

            return_dict['z_entropy'] = z_dist.entropy().mean().item()
            return_dict['zz_kl_tensor'] = kld_zz_nograd.cpu().numpy()
            return_dict['zz_lld_tensor'] = q_minus_p_nograd.cpu().numpy()
            return_dict['zz_kld'] = kld_zz_nograd[:-1].mean().item()
            return_dict['zz_kl_loss'] = \
                infobot_beta * infobot_kl_coeff * kld_zz_nograd[:-1].mean().item()

            # Tracking default actions
            obs_default = obs_base.clone()
            z_prior_samples = z_prior.sample()
            obs_default.update({
                'z_latents': z_prior_samples,
            })
            _, default_action_dist, _, _ = \
                self.actor_critic.get_action_dist(
                    inputs=obs_default,
                    rnn_hxs=rollouts.recurrent_hidden_states[0].view(
                        -1, self.actor_critic.recurrent_hidden_state_size),
                    masks=rollouts.masks[:-1].view(-1, 1))
            _probs = default_action_dist.probs
            default_action_probs = _probs.view(
                num_steps, num_processes, *_probs.shape[1:])
            # default_action_dist = FixedCategorical(probs=default_action_probs)
            return_dict['default_action_probs'] = default_action_probs

        if ib_adaptive:
            # eff_coeff = infobot_beta * infobot_kl_coeff
            eff_coeff = infobot_beta * ic_adaptive_wts
            _min = ic_adaptive_wts.new_ones(1) * min_ib_coeff
            eff_coeff = torch.max(eff_coeff, _min)
            ib_kl_objective = eff_coeff * kld_zz
        else:
            ib_kl_objective = infobot_beta * infobot_kl_coeff * kld_zz

        return obs_base, ib_kl_objective, default_action_dist, return_dict

    def kl_exploration_bonus(
        self,
        agent_pos,
        rollouts,
        bonus_z_encoder,
        bonus_type,
        bonus_normalization,
        visit_count,
        heuristic_ds,
        heuristic_coeff,
        b_args,
        ic_info,
    ):
        masks = rollouts.masks

        # Part 1: Z-KL values
        if bonus_type != 'count':
            # # Part 1: Z-KL values
            kld = bonus_kl_forward(
                obs=rollouts.obs.copy(),
                b_args=b_args,
                bonus_z_encoder=bonus_z_encoder,
                bonus_type=bonus_type,
                masks=masks,
                bonus_normalization=bonus_normalization,
            )
        else:
            kld = torch.zeros_like(masks)

        # # Ignore Z-KL for Z sampeld after last time step
        # kld = kld[:-1]

        # Part 2: Visitation count
        r_t_bonus = torch.zeros_like(masks)
        agent_pos = torch.from_numpy(agent_pos).to(masks.device)
        plus_masks = torch.eq(torch.cat([masks[0:1], masks[:-1]], 0), 1)

        visitation_x = agent_pos[:, :, 0:1].masked_select(
            torch.eq(plus_masks, 1)).cpu().numpy()
        visitation_y = agent_pos[:, :, 1:2].masked_select(
            torch.eq(plus_masks, 1)).cpu().numpy()
        kl_values = kld.masked_select(
            torch.eq(plus_masks, 1)).cpu().numpy()

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

        visit_count = torch.from_numpy(visit_count.astype('float32')).to(masks.device)
        inv_sqrt_count = 1 / torch.sqrt(visit_count)

        # Noise added to visit count
        bonus_noise = self.bonus_noise_grid[visitation_x, visitation_y]
        bonus_noise = torch.from_numpy(bonus_noise).to(masks.device)

        heuristic_ds = torch.from_numpy(heuristic_ds).float().to(masks.device)
        coeff = (heuristic_ds * heuristic_coeff) \
            + ((1-heuristic_ds) * self.bonus_reward.beta)
        # count_bonus = self.bonus_reward.beta * inv_sqrt_count
        count_bonus = coeff * inv_sqrt_count
        bonus_noise_tensor = torch.zeros_like(count_bonus)
        bonus_noise_tensor.masked_scatter_(plus_masks.squeeze(-1), bonus_noise)
        count_bonus += bonus_noise_tensor

        r_t_bonus += count_bonus.unsqueeze(-1)

        return r_t_bonus, kld
