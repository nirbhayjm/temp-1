from typing import Dict, Optional, Tuple

import numpy as np
import os, sys

import gym
import torch
import torch.optim as optim
import distributions
import torch.distributions as ds

from storage import RolloutStorage, DictObs
import utilities.utilities as utils
# import utilities.eval_utils as eval_utils
from utilities.signal import GracefulSignalHandler
from utilities.eval_utils import eval_ib_kl, eval_success

from train_logger import TrainLogger


def flatten_batch_dims(*args):
    flattened_args = []

    for ten in args:
        eff_dim = ten.shape[0] * ten.shape[1]
        flattened_args.append(ten.reshape((eff_dim,) + ten.shape[2:]))

    return flattened_args


def dict_stack_helper(list_of_dicts):
    return {key:np.stack([item[key] for item in list_of_dicts], 0) \
        for key, val in list_of_dicts[0].items() if type(val) != str}

def repalce_mission_with_omega(
    obs_input: DictObs, omega: torch.Tensor) -> DictObs:
    # obs = copy.deepcopy(obs_input)
    obs = obs_input.copy()
    if 'mission' in obs.keys():
        obs.pop('mission')
    obs.update({'omega': omega.clone()})
    return obs

def repalce_omega_with_z(
    obs_input: DictObs, z_latent: torch.Tensor) -> DictObs:
    # obs = copy.deepcopy(obs_input)
    obs = obs_input.copy()
    if 'omega' in obs.keys():
        obs.pop('omega')
    obs.update({'z_latent': z_latent.clone()})
    return obs

def replace_goal_vector_with_z(
    obs_input: DictObs, z_latent: torch.Tensor) -> DictObs:
    # obs = copy.deepcopy(obs_input)
    obs = obs_input.copy()
    if 'goal_vector' in obs.keys():
        obs.pop('omega')
    obs.update({'z_latent': z_latent.clone()})
    return obs

class Trainer(object):
    def __init__(
        self,
        args: Dict,
        train_envs,
        val_envs,
        vis_env,
        actor_critic,
        options_policy,
        options_decoder,
        trajectory_encoder,
        trajectory_optim,
        z_encoder,
        b_args,
        agent,
        args_state,
        rollouts: RolloutStorage,
        device: torch.device,
        num_processes_eff: int,
    ):
        self.args = args
        self.train_envs = train_envs
        self.val_envs = val_envs
        self.actor_critic = actor_critic
        self.options_decoder = options_decoder
        self.trajectory_encoder = trajectory_encoder
        self.trajectory_optim = trajectory_optim
        self.options_policy = options_policy
        self.z_encoder = z_encoder
        self.b_args = b_args
        self.agent = agent
        self.args_state = args_state
        self.rollouts = rollouts
        self.num_processes_eff = num_processes_eff
        self.device = device
        NUM_BATCHES_PER_EPOCH = 100
        self.num_batches_per_epoch = NUM_BATCHES_PER_EPOCH
        self.continuous_state_space = False
        if self.args.env_name in ['mountain-car', 'acrobat']:
            self.continuous_state_space = True

        self.logger = TrainLogger(
            args=args,
            vis_env=vis_env,
            val_envs=val_envs,
            device=device,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_processes_eff=self.num_processes_eff,
            continuous_state_space=self.continuous_state_space,
        )

        self.omega_dim_growth_ratio = 1.5 # from VALOR
        self.omega_dim_ll_threshold = np.log(
            self.args.omega_traj_ll_theta) # from VALOR
        self.min_omega_dim = min(2, self.args.omega_option_dims)
        if self.args.model == 'cond' \
        or self.args.hier_mode == 'infobot-supervised' \
        or self.args.option_space == 'continuous':
            self.omega_dim_current = self.args.omega_option_dims
        elif hasattr(self.args, 'omega_dim_current'):
            self.omega_dim_current = self.args.omega_dim_current
        elif self.args.use_omega_dim_curriculum and self.args.hier_mode != 'transfer':
            self.omega_dim_current = self.min_omega_dim
        else:
            self.omega_dim_current = self.args.omega_option_dims

        if self.args.reset_adaptive:
            print("Using adaptive reset, setting initial reset_prob to 1.0")
            reset_probs = [1.0 for _ in range(self.args.num_processes)]
            self.train_envs.modify_attr('reset_prob', reset_probs)

        self.total_time_steps = 0

        self.to(device)

    def to(self, device):
        self.rollouts.to(device)
        self.actor_critic.to(device)
        if self.trajectory_encoder is not None:
            self.trajectory_encoder.to(device)
            self.trajectory_optim = optim.Adam(
                self.trajectory_encoder.parameters(),
                lr=self.args.lr, eps=self.args.eps)

        if hasattr(self.agent, 'options_decoder'):
            self.agent.options_decoder.to(device)

        if self.options_policy is not None:
            self.options_policy.to(device)

        if self.z_encoder is not None:
            self.z_encoder.to(device)

        self.agent.init_optims()

    def on_train_start(self):
        if self.args.model == 'hier':
            self.do_sampling = True
        elif self.args.model == 'cond':
            self.do_sampling = False

        if self.args.infobot_auto_kl and self.omega_dim_current == self.min_omega_dim:
            self.do_z_sampling = False
        else:
            self.do_z_sampling = bool(self.args.z_stochastic)

        if self.args.infobot_beta > 0.0 and self.args.use_infobot:
            assert bool(self.args.z_stochastic) == True

    def on_episode_start(self):
        self.actor_critic.train()

        # obs = train_envs.reset()
        reset_output = self.train_envs.reset()
        obs = reset_output[:, 0]
        info = reset_output[:, 1]
        obs = dict_stack_helper(obs)
        info = dict_stack_helper(info)

        if not self.continuous_state_space:
            self.visit_count = [np.ones(self.num_processes_eff)]
            self.agent_pos = np.zeros(
                [self.args.num_steps + 1, self.num_processes_eff, 2], dtype='int')
            self.agent_pos[0] = info['agent_pos']
            info['pos_velocity'] = None
        else:
            self.agent_pos = None
            if self.args.env_name == 'mountain-car':
                info['pos_velocity'] = obs['pos-velocity']
            else:
                info['pos_velocity'] = np.zeros((self.num_processes_eff, 2))

        self.heuristic_ds = np.zeros(
            [self.args.num_steps + 1, self.num_processes_eff], dtype='int')

        # [obs] = flatten_batch_dims(obs)
        obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
            for key, obs_i in obs.items()})

        if self.args.model == 'cond':
            omega_option = None
            q_dist_ref = None
            options_rhx = None
            ib_rnn_hx = None
            self.rollouts.obs[0].copy_(obs)
            return omega_option, obs, q_dist_ref, ib_rnn_hx, \
                options_rhx, info

        if self.args.use_infobot:
            ib_rnn_hx = self.rollouts.recurrent_hidden_states.new_zeros(
                self.num_processes_eff,
                self.actor_critic.encoder_recurrent_hidden_state_size)

            if self.args.hier_mode == 'infobot-supervised':
                omega_option = None
                q_dist_ref = None
                options_rhx = None
                self.rollouts.obs[0].copy_(obs)

                z_latent, z_log_prob, z_dist, ib_rnn_hx = \
                self.actor_critic.encoder_forward(
                    obs=obs,
                    rnn_hxs=ib_rnn_hx,
                    masks=self.rollouts.masks[0],
                    do_z_sampling=True,
                )
                self.rollouts.insert_z_latent(
                    z_latent, z_log_prob, z_dist, ib_rnn_hx)

                obs = replace_goal_vector_with_z(obs, z_latent)

                return omega_option, obs, q_dist_ref, ib_rnn_hx, \
                    options_rhx, info
        else:
            ib_rnn_hx = None


        if self.args.option_space == 'continuous':
            option_log_probs = None
            if self.args.hier_mode == 'default':
                omega_option, q_dist, _, ldj = self.options_decoder(
                    obs, do_sampling=self.do_sampling)

            elif self.args.hier_mode == 'vic':
                ldj = 0.0
                _shape = (self.num_processes_eff, self.args.omega_option_dims)
                _loc = torch.zeros(*_shape).to(self.device)
                _scale = torch.ones(*_shape).to(self.device)
                # if self.omega_dim_current < self.args.omega_option_dims:
                #     _scale[:, self.omega_dim_current:].fill_(1e-3)
                q_dist = ds.normal.Normal(loc=_loc, scale=_scale)
                if self.do_sampling:
                    omega_option = q_dist.rsample()
                else:
                    omega_option = q_dist.mean

                if self.args.ic_mode == 'diyan':
                    _shape_t = (self.args.num_steps + 1, *_shape)
                    _loc_t = torch.zeros(*_shape_t).to(self.device)
                    _scale_t = torch.ones(*_shape_t).to(self.device)
                    # if self.omega_dim_current < self.args.omega_option_dims:
                    #     _scale_t[:, :, self.omega_dim_current:].fill_(1e-3)
                    q_dist_ref = ds.normal.Normal(loc=_loc_t, scale=_scale_t)
                else:
                    q_dist_ref = q_dist

                if self.args.use_infobot:
                    obs_omega = repalce_mission_with_omega(
                        obs, omega_option)
                    z_latent, z_log_prob, z_dist, ib_rnn_hx = \
                        self.actor_critic.encoder_forward(
                            obs=obs_omega,
                            rnn_hxs=ib_rnn_hx,
                            masks=self.rollouts.masks[0],
                            do_z_sampling=self.do_z_sampling)

            elif self.args.hier_mode == 'transfer':
                # omega_option, q_dist, _, ldj = self.options_policy(
                #     obs, do_sampling=self.do_sampling)
                omega_option = None
                q_dist_ref = None
            else:
                raise ValueError

        else:
            ldj = 0.0
            if self.args.hier_mode == 'default':
                with torch.no_grad():
                    option_discrete, q_dist, option_log_probs = self.options_decoder(
                        obs, do_sampling=self.do_sampling)

                    if self.args.use_infobot:
                        raise NotImplementedError

            elif self.args.hier_mode == 'vic':
                with torch.no_grad():
                    _shape = (self.num_processes_eff, self.args.omega_option_dims)
                    uniform_probs = torch.ones(*_shape).to(self.device)
                    if self.omega_dim_current < self.args.omega_option_dims:
                        uniform_probs[:, self.omega_dim_current:].fill_(0)
                    uniform_probs = uniform_probs / uniform_probs.sum(-1, keepdim=True)
                    q_dist = distributions.FixedCategorical(probs=uniform_probs)
                    option_discrete = q_dist.sample()
                    # option_log_probs = q_dist.log_probs(option_discrete)

                    if self.args.ic_mode == 'diyan':
                        _shape_t = (self.args.num_steps + 1, *_shape)
                        uniform_probs = torch.ones(*_shape_t).to(self.device)
                        if self.omega_dim_current < self.args.omega_option_dims:
                            uniform_probs[:, :, self.omega_dim_current:].fill_(0)
                        uniform_probs = uniform_probs / uniform_probs.sum(-1, keepdim=True)
                        q_dist_ref = distributions.FixedCategorical(probs=uniform_probs)
                    else:
                        q_dist_ref = q_dist

                    if self.args.use_infobot:
                        omega_one_hot = torch.eye(self.args.omega_option_dims)[option_discrete]
                        omega_one_hot = omega_one_hot.float().to(self.device)
                        obs_omega = repalce_mission_with_omega(
                            obs, omega_one_hot)
                        z_latent, z_log_prob, z_dist, ib_rnn_hx = \
                            self.actor_critic.encoder_forward(
                                obs=obs_omega,
                                rnn_hxs=ib_rnn_hx,
                                masks=self.rollouts.masks[0],
                                do_z_sampling=self.do_z_sampling)

            elif self.args.hier_mode in ['transfer', 'bonus']:
                omega_option = None
                q_dist_ref = None

            else:
                raise ValueError

            if self.args.hier_mode != 'transfer':
                option_np = option_discrete.squeeze(-1).cpu().numpy()
                option_one_hot = np.eye(self.args.omega_option_dims)[option_np]
                omega_option = torch.from_numpy(option_one_hot).float().to(self.device)

        if self.args.hier_mode == 'transfer':
            obs_base = obs
            if self.args.use_infobot:
                raise NotImplementedError
            else:
                pass
        else:
            if self.args.use_infobot:
                obs_base = repalce_omega_with_z(obs, z_latent)
                self.rollouts.insert_option(omega_option)
                self.rollouts.insert_z_latent(z_latent, z_log_prob, z_dist, ib_rnn_hx)
            else:
                obs_base = repalce_mission_with_omega(obs, omega_option)
                self.rollouts.insert_option(omega_option)

        if self.args.hier_mode == 'transfer':
            options_rhx = torch.zeros(self.num_processes_eff,
                self.options_policy.recurrent_hidden_state_size).to(self.device)
        else:
            options_rhx = None

        # self.omega_option = omega_option
        # self.obs_base = obs_base
        self.rollouts.obs[0].copy_(obs)

        return omega_option, obs_base, q_dist_ref, ib_rnn_hx, \
            options_rhx, info

    def forward_step(self, step, omega_option, obs_base, ib_rnn_hxs, options_rhx):
        # Sample options if applicable
        if self.args.hier_mode == 'transfer':
            with torch.no_grad():
                if step % self.args.num_option_steps == 0:
                    omega_option = None
                    previous_options_rhx = options_rhx
                    option_value, omega_option, option_log_probs, options_rhx = \
                        self.options_policy.act(
                            inputs=obs_base,
                            rnn_hxs=options_rhx,
                            masks=self.rollouts.masks[step])
                    if self.args.option_space == 'discrete':
                        omega_option = omega_option.squeeze(-1)
                        omega_option = torch.eye(self.args.omega_option_dims)\
                            .to(self.device)[omega_option]
                    self.rollouts.insert_option_t(
                        step=step,
                        omega_option_t=omega_option,
                        option_log_probs=option_log_probs,
                        option_value=option_value,
                        options_rhx=previous_options_rhx)
                obs_base = repalce_mission_with_omega(obs_base, omega_option)

        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = \
                self.actor_critic.act(
                    inputs=obs_base,
                    rnn_hxs=self.rollouts.recurrent_hidden_states[step],
                    masks=self.rollouts.masks[step])

        # Take actions, observe reward and next obs
        # cpu_actions = action.view(
        #     (self.args.num_processes, self.args.num_agents)).cpu().numpy()
        cpu_actions = action.view(-1).cpu().numpy()

        # obs, reward, _, info = self.train_envs.step(cpu_actions + 1)
        obs, reward, _, info = self.train_envs.step(cpu_actions)

        obs = dict_stack_helper(obs)
        obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
            for key, obs_i in obs.items()})

        if self.args.hier_mode == 'transfer' or self.args.model == 'cond':
            obs_base = obs
        else:
            if self.args.use_infobot:
                if self.args.hier_mode == 'infobot-supervised':
                    z_latent, z_log_prob, z_dist, ib_rnn_hxs = \
                        self.actor_critic.encoder_forward(
                            obs=obs,
                            rnn_hxs=ib_rnn_hxs,
                            masks=self.rollouts.masks[step],
                            do_z_sampling=True)

                    obs_base = replace_goal_vector_with_z(obs, z_latent)
                else:
                    # Sample next z_t
                    obs_omega = repalce_mission_with_omega(
                        obs, omega_option)
                    z_latent, z_log_prob, z_dist, ib_rnn_hxs = \
                        self.actor_critic.encoder_forward(
                            obs=obs_omega,
                            rnn_hxs=ib_rnn_hxs,
                            masks=self.rollouts.masks[step],
                            do_z_sampling=self.do_z_sampling)

                    obs_base = repalce_omega_with_z(obs, z_latent)
                self.rollouts.insert_z_latent(
                    z_latent=z_latent,
                    z_logprobs=z_log_prob,
                    z_dist=z_dist,
                    ib_enc_hidden_states=ib_rnn_hxs)
            else:
                obs_base = repalce_mission_with_omega(obs, omega_option)

        done = np.stack([item['done'] for item in info], 0)
        if 'is_heuristic_ds' in info[0].keys():
            is_heuristic_ds = np.stack([item['is_heuristic_ds'] for item in info], 0)
            self.heuristic_ds[step + 1] = is_heuristic_ds

        if not self.continuous_state_space:
            curr_pos = np.stack([item['agent_pos'] for item in info], 0)
            curr_dir = np.stack([item['agent_dir'] for item in info], 0)
            visit_count = np.stack([item['visit_count'] for item in info], 0)
            self.agent_pos[step + 1] = curr_pos

            # if 'current_room' in info[0]:
            #     self.current_room = np.stack(
            #         [item['current_room'] for item in info], 0)
            self.visit_count.append(visit_count)
            pos_velocity = None
        else:
            curr_pos = None
            curr_dir = None
            if self.args.env_name == 'mountain-car':
                pos_velocity = obs['pos-velocity']
            else:
                pos_velocity = np.zeros((self.num_processes_eff, 2))

        # [obs, reward] = utils.flatten_batch_dims(obs,reward)
        # print(step, done)
        # Extract the done flag from the info
        # done = np.concatenate([info_['done'] for info_ in info],0)

        # if step == self.args.num_steps - 1:
        #     s_extract = lambda key_: np.array(
        #         [item[key_] for item in info])
        #     success_train = s_extract('success').astype('float')
        #     goal_index = s_extract('goal_index')
        #     success_0 = success_train[goal_index == 0]
        #     success_1 = success_train[goal_index == 1]
        #     # spl_train = s_extract('spl_values')
        #     # Shape Assertions

        reward = torch.from_numpy(reward[:,np.newaxis]).float()
        # episode_rewards += reward
        cpu_reward = reward
        reward = reward.to(self.device)
        # reward = torch.from_numpy(reward).float()

        not_done = np.logical_not(done)
        self.total_time_steps += not_done.sum()

        masks = torch.from_numpy(not_done.astype('float32')).unsqueeze(1)
        masks = masks.to(self.device)

        for key in obs.keys():
            if obs[key].dim() == 5:
                obs[key] *= masks.type_as(obs[key])\
                    .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            elif obs[key].dim() == 4:
                obs[key] *= masks.type_as(obs[key]).unsqueeze(-1).unsqueeze(-1)
            elif obs[key].dim() == 1:
                obs[key] *= masks.type_as(obs[key]).squeeze(1)
            else:
                obs[key] *= masks.type_as(obs[key])

        self.rollouts.insert(obs, recurrent_hidden_states, action,
            action_log_prob, value, reward, masks)

        return obs_base, omega_option, ib_rnn_hxs, options_rhx, cpu_reward, \
            curr_pos, curr_dir, pos_velocity, not_done

    def on_episode_end(
        self,
        iter_id,
        epoch_id,
        obs_base,
        omega_option,
        q_dist_ref,
        options_rhx,
        ib_rnn_hxs,
    ):
        with torch.no_grad():
            if self.args.hier_mode == 'transfer':
                option_value = self.options_policy.get_value(
                    inputs=obs_base,
                    rnn_hxs=options_rhx,
                    masks=self.rollouts.masks[-1])
                # if self.args.option_space == 'discrete':
                #     omega_option = omega_option.squeeze(-1)
                #     omega_option = torch.eye(self.args.omega_option_dims)\
                #         .to(self.device)[omega_option]
                self.rollouts.insert_next_option_value(option_value)
                obs_base = repalce_mission_with_omega(obs_base, omega_option)
                # self.rollouts.insert_option_t(
                #     omega_option_t=omega_option,
                #     option_log_probs=option_log_probs,
                #     option_value=option_value)

            next_value = self.actor_critic.get_value(
                inputs=obs_base,
                rnn_hxs=self.rollouts.recurrent_hidden_states[-1],
                masks=self.rollouts.masks[-1],
            ).detach()
            self.rollouts.insert_next_value(next_value)

        anneal_coeff = utils.kl_coefficient_curriculum(
            iter_id=iter_id,
            iters_per_epoch=self.num_batches_per_epoch,
            start_after_epochs=self.args.kl_anneal_start_epochs,
            linear_growth_epochs=self.args.kl_anneal_growth_epochs,
        )

        if self.do_z_sampling == False:
            infobot_coeff = 0
        else:
            infobot_coeff = utils.kl_coefficient_curriculum(
                iter_id=iter_id,
                iters_per_epoch=self.num_batches_per_epoch,
                start_after_epochs=self.args.infobot_kl_start,
                linear_growth_epochs=self.args.infobot_kl_growth,
            )

        min_ib_coeff = min(self.args.infobot_beta_min, self.args.infobot_beta)
        if self.args.infobot_beta > 0:
            infobot_coeff = max(infobot_coeff, min_ib_coeff / self.args.infobot_beta)
            if self.do_z_sampling == False:
                infobot_coeff = 0

        q_start_flag = utils.q_start_curriculum(
            iter_id=iter_id,
            iters_per_epoch=self.num_batches_per_epoch,
            start_after_epochs=self.args.q_start_epochs,
        )

        if self.continuous_state_space:
            visit_count = None
        else:
            visit_count = np.stack(self.visit_count, 0)

        if self.args.algo == 'a2c' or self.args.algo == 'acktr':
            if self.args.model == 'hier':
                ic_kwargs = {
                    'ic_mode': self.args.ic_mode,
                    # 'p_dist': p_dist,
                    'q_dist': q_dist_ref,
                    # 'log_det_j': ldj,
                    'log_det_j': 0,
                    'q_start_flag': q_start_flag,
                    'kl_coeff': self.args.hr_model_kl_coeff,
                    'kl_optim_mode': self.args.kl_optim_mode,
                    'reweight_by_omega_ll': self.args.reweight_by_omega_ll,
                    'anneal_coeff': anneal_coeff,
                    'option_space': self.args.option_space,
                    'traj_encoder_input': self.args.traj_encoder_input,
                    'trajectory_encoder': self.trajectory_encoder,
                }

                ib_kwargs = {
                    'infobot_beta': self.args.infobot_beta,
                    'infobot_kl_coeff': infobot_coeff,
                    'kl_optim_mode': self.args.ib_kl_mode,
                    'z_dist_type': 'gaussian',
                    'ib_adaptive': self.args.ib_adaptive,
                    'min_ib_coeff': min_ib_coeff,
                }

                # vic_only = False
                # if self.args.hier_mode == 'vic':
                #     vic_only = True
                use_intrinsic_control = self.args.hier_mode != 'transfer' and \
                    self.args.hier_mode != 'infobot-supervised'
                value_loss, action_loss, dist_entropy,\
                action_log_probs_mean, ic_info, option_info = \
                    self.agent.update(
                        rollouts=self.rollouts,
                        # vic_only=vic_only,
                        option_space=self.args.option_space,
                        # option_log_probs=option_log_probs,
                        use_intrinsic_control=use_intrinsic_control,
                        hier_mode=self.args.hier_mode,
                        ic_kwargs=ic_kwargs,
                        trajectory_encoder=self.trajectory_encoder,
                        trajectory_optim=self.trajectory_optim,
                        traj_enc_loss_coeff=self.args.traj_enc_loss_coeff,
                        use_ib=self.args.use_infobot,
                        ib_kwargs=ib_kwargs,
                        agent_pos=self.agent_pos,
                        bonus_z_encoder=self.z_encoder,
                        b_args=self.b_args,
                        bonus_type=self.args.bonus_type,
                        bonus_normalization=self.args.bonus_normalization,
                        heuristic_ds=self.heuristic_ds,
                        heuristic_coeff=self.args.bonus_heuristic_beta,
                        visit_count=visit_count,
                    )

            else:
                # Conditional model
                value_loss, action_loss, dist_entropy,\
                action_log_probs_mean, ic_info, option_info = \
                    self.agent.update(
                        self.rollouts,
                        hier_mode=self.args.hier_mode,
                        use_intrinsic_control=False,
                        use_ib=False,
                        agent_pos=self.agent_pos,
                        bonus_z_encoder=self.z_encoder,
                        b_args=self.b_args,
                        bonus_type=self.args.bonus_type,
                        bonus_normalization=self.args.bonus_normalization,
                        heuristic_ds=self.heuristic_ds,
                        heuristic_coeff=self.args.bonus_heuristic_beta,
                        visit_count=visit_count,
                    )

            ic_info.update({
                'anneal_coeff': anneal_coeff,
                'infobot_coeff': infobot_coeff,
                'q_start_flag': q_start_flag,
            })

        else:
            raise ValueError("Unknown algo: {}".format(self.args.algo))

        # Adaptive reset based on IC performance (empowerment)
        if self.args.reset_adaptive:
            # reset_probs = [self.args.reset_prob \
            #     for _ in range(self.args.num_processes)]
            # if ic_info['empowerment_value'] > 0.1
            new_reset_probs = (ic_info['r_adaptive_wts'] <= 0).astype('float')
            self.train_envs.modify_attr('reset_prob', new_reset_probs)
            # new_probs = self.train_envs.get_attr('reset_prob')
            # if new_probs.min() == 0:
            #     print("Did not reset {}".format(new_probs))

        if self.args.model == 'cond' \
        and self.args.hier_mode == 'bonus' \
        and iter_id % (self.args.log_interval) == 0:
            with torch.no_grad():
                success_val, max_room_id, vis_info = eval_success(
                    args=self.args,
                    val_envs=self.val_envs,
                    vis_env=self.logger.vis_env,
                    actor_critic=self.actor_critic,
                    b_args=self.b_args,
                    bonus_beta=self.args.bonus_beta,
                    bonus_type=self.args.bonus_type,
                    bonus_z_encoder=self.z_encoder,
                    bonus_normalization=self.args.bonus_normalization,
                    device=self.device,
                    num_episodes=self.args.num_eval_episodes,
                )
                self.logger.update_visdom_success_plot(
                    iter_id, success_val, max_room_id, vis_info)

        # Checkpoint saving
        if iter_id % (self.args.save_interval * self.num_batches_per_epoch) \
            == 0 and self.args.save_dir != "":
            # print("[WARN] DEBUG MODE: NOT SAVING CHECKPOINTS!")
            self.save_checkpoint(iter_id=iter_id, epoch_id=epoch_id)

        eval_info = {}
        if self.args.use_infobot \
        and iter_id % (self.args.heatmap_interval) == 0\
        and bool(self.args.recurrent_policy) == False \
        and bool(self.args.recurrent_encoder) == False \
        and self.args.is_pomdp_env == False\
        and self.continuous_state_space == False:
            with torch.no_grad():
                eval_info = eval_ib_kl(
                    args=self.args,
                    vis_env=self.logger.vis_env,
                    actor_critic=self.actor_critic,
                    device=self.device,
                    omega_dim_current=self.omega_dim_current,
                    num_samples=200)
                # eval_info['z_kl_grid'] = z_kl_grid

        return value_loss, action_loss, dist_entropy,\
            action_log_probs_mean, ic_info, option_info, eval_info

    def train(self, start_iter, num_epochs):
        """Train loop"""

        print("="*36)
        print("Trainer initialized! Training information:")
        print("\t# of epochs: {}".format(num_epochs))
        # print("\t# of train envs: {}".format(len(self.train_envs)))
        print("\tnum_processes: {}".format(self.args.num_processes))
        print("\tnum_agents: {}".format(self.args.num_agents))
        print("\tIterations per epoch: {}".format(self.num_batches_per_epoch))
        print("="*36)

        def batch_iterator(start_idx):
            idx = start_idx
            epoch_id = idx // self.num_batches_per_epoch
            for _ in range(num_epochs):
                for _ in range(self.num_batches_per_epoch):
                    yield epoch_id, idx, None
                    idx += 1
                epoch_id += 1

        # signal_handler = GracefulSignalHandler()
        self.on_train_start()
        self.logger.on_train_start()

        for epoch_id, iter_id, _ in batch_iterator(start_iter):
            omega_option, obs_base, q_dist_ref, ib_rnn_hxs, options_rhx, info = \
                self.on_episode_start()

            self.logger.on_iter_start(info=info)

            for step in range(self.args.num_steps):
                # Step
                obs_base, omega_option, ib_rnn_hxs, options_rhx, cpu_reward, \
                curr_pos, curr_dir, pos_velocity, not_done = \
                    self.forward_step(
                        step=step,
                        omega_option=omega_option,
                        obs_base=obs_base,
                        ib_rnn_hxs=ib_rnn_hxs,
                        options_rhx=options_rhx,
                    )

                self.logger.update_at_step(
                    step=step,
                    reward_t=cpu_reward,
                    agent_pos=curr_pos,
                    agent_dir=curr_dir,
                    pos_velocity=pos_velocity,
                    not_done=not_done,
                )

            value_loss, action_loss, dist_entropy, action_log_probs_mean, \
                ic_info, option_info, eval_info = \
                self.on_episode_end(
                    iter_id=iter_id,
                    epoch_id=epoch_id,
                    obs_base=obs_base,
                    omega_option=omega_option,
                    q_dist_ref=q_dist_ref,
                    options_rhx=options_rhx,
                    ib_rnn_hxs=ib_rnn_hxs,
                )

            # if hasattr(self, 'current_room'):
            #     current_room = self.current_room
            # else:
            #     current_room = None

            traj_enc_ll_average, empowerment_avg = self.logger.on_iter_end(
                start_iter=start_iter,
                iter_id=iter_id,
                total_time_steps=self.total_time_steps,
                rollouts=self.rollouts,
                omega_option=omega_option,
                omega_dim_current=self.omega_dim_current,
                omega_dim_ll_threshold=self.omega_dim_ll_threshold,
                value_loss=value_loss,
                action_loss=action_loss,
                dist_entropy=dist_entropy,
                action_log_probs_mean=action_log_probs_mean,
                ic_info=ic_info,
                option_info=option_info,
                eval_info=eval_info,
            )

            if self.args.use_omega_dim_curriculum:
                # Keep this same as M_AVG_WIN_SIZE
                if iter_id % (self.args.omega_curr_win_size // 2) == 0:
                    new_omega_dim = utils.omega_dims_curriculum(
                        traj_enc_ll=traj_enc_ll_average,
                        threshold=self.omega_dim_ll_threshold,
                        current_omega=self.omega_dim_current,
                        max_omega=self.args.omega_option_dims,
                        growth_ratio=self.omega_dim_growth_ratio,
                    )
                    if new_omega_dim != self.omega_dim_current:
                        self.save_checkpoint(iter_id, epoch_id,
                            suffix="_omega{:02d}".format(self.omega_dim_current))
                        self.omega_dim_current = new_omega_dim
                    self.args.omega_dim_current = self.omega_dim_current

            if self.args.infobot_auto_kl:
                EMPOWERMENT_LOWER_LIMIT = 0.5
                # EMPOWERMENT_LOWER_LIMIT = -0.3
                if self.do_z_sampling == False \
                and self.args.z_stochastic == True \
                and iter_id % (self.args.omega_curr_win_size // 2) == 0 \
                and empowerment_avg > EMPOWERMENT_LOWER_LIMIT:
                    self.do_z_sampling = True
                    print("Z deterministic -> stochastic!")
                    # print("IB KL Start old: {}".format(self.args.infobot_kl_start))
                    self.args.infobot_kl_start = int(epoch_id) + self.args.infobot_kl_start
                    print("IB KL Start new: {}".format(self.args.infobot_kl_start))

            # Rollouts cleanup after monte-carlo update
            # NOTE: This means no TD updates supported
            self.rollouts.after_update()
            self.rollouts.reset_storage()

            # if signal_handler.kill_now:
            #     if os.getpid() == signal_handler.parent_pid:
            #         _time_stamp = signal_handler.get_time_str()
            #         self.logger.viz.viz.text("Process {}, exit time: {}".format(
            #             signal_handler.parent_pid, _time_stamp))
            #         print("Time of exit: {}".format(_time_stamp))
            #         print("Exited gracefully!")
            #     sys.exit(0)
            pass

    def save_checkpoint(self, iter_id, epoch_id, suffix=""):
        # Save checkpoint
        self.args.epoch_id = epoch_id
        self.args.iter_id = iter_id
        os.makedirs(self.args.save_dir, exist_ok=True)
        save_path = os.path.join(
            self.args.save_dir,
            self.args.algo + "_{:04d}{}.vd".format(int(epoch_id), suffix),
        )

        # A really ugly way to save a model to CPU
        # save_model = actor_critic
        # if args.cuda:
        #     save_model = copy.deepcopy(actor_critic).cpu()

        save_dict = {
          'model': self.actor_critic.state_dict(),
          'options_decoder': self.options_decoder.state_dict(),
          'params': vars(self.args),
          # 'policy_kwargs' : policy_kwargs,
          # 'train_kwargs' : train_kwargs
          'args_state': self.args_state,
        }

        if self.args.model == 'hier':
            if self.trajectory_encoder is not None:
                save_dict['trajectory_encoder'] = \
                    self.trajectory_encoder.state_dict()
            if self.trajectory_optim is not None:
                save_dict['trajectory_optim'] = \
                    self.trajectory_optim.state_dict()
            if hasattr(self.agent, 'options_policy_optim'):
                save_dict['options_policy_optim'] = \
                    self.agent.options_policy_optim.state_dict()
            save_dict['actor_critic_optim'] = \
                self.agent.actor_critic_optim.state_dict()
        else:
            save_dict['optimizer'] = \
                self.agent.optimizer.state_dict()

        print("Currently on visdom env: {}".format(self.args.visdom_env_name))
        print("Saving checkpoint:", save_path)
        torch.save(save_dict, save_path)
