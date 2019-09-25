from typing import Dict, Optional, Tuple

import numpy as np
import os, sys, time
from collections import deque

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

from train_logger import TrainLoggerTD


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


class TrainerTD(object):
    def __init__(
        self,
        args: Dict,
        train_envs,
        val_envs,
        vis_env,
        actor_critic,
        options_policy,
        options_decoder,
        # option_prior,
        # option_prior_optim,
        # trajectory_encoder,
        # trajectory_optim,
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
        self.vis_env = vis_env
        self.actor_critic = actor_critic
        self.options_decoder = options_decoder
        # self.option_prior = option_prior
        # self.option_prior_optim = option_prior_optim
        # self.trajectory_encoder = trajectory_encoder
        # self.trajectory_optim = trajectory_optim
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

        self.logger = TrainLoggerTD(
            args=args,
            vis_env=vis_env,
            # val_envs=val_envs,
            # device=device,
            # num_batches_per_epoch=self.num_batches_per_epoch,
            # num_processes_eff=self.num_processes_eff,
            # continuous_state_space=True,
        )

        self.next_val_after = 1 * self.args.val_interval

        if self.args.reset_adaptive:
            print("Using adaptive reset, setting initial reset_prob to 1.0")
            reset_probs = [1.0 for _ in range(self.args.num_processes)]
            self.train_envs.modify_attr('reset_prob', reset_probs)

        # self.option_prior_requires_grad = self.check_requires_grad(self.option_prior)

        self.to(device)

    def check_requires_grad(self, net):
        requires_grad = False
        for param in net.parameters():
            if param.requires_grad:
                requires_grad = True
        return requires_grad

    def to(self, device):
        self.rollouts.to(device)
        self.actor_critic.to(device)
        # if self.trajectory_encoder is not None:
        #     self.trajectory_encoder.to(device)
        # self.option_prior.to(device)
        if self.options_policy is not None:
            self.options_policy.to(device)
        if self.z_encoder is not None:
            self.z_encoder.to(device)
        self.agent.init_optims()

    def train(self, total_training_steps, start_iter):
        """Train loop"""

        print("="*36)
        print("Trainer initialized! Training information:")
        print("\t# of total_training_steps: {}".format(total_training_steps))
        # print("\t# of train envs: {}".format(len(self.train_envs)))
        print("\tnum_processes: {}".format(self.args.num_processes))
        print("\tnum_agents: {}".format(self.args.num_agents))
        # print("\tIterations per epoch: {}".format(self.num_batches_per_epoch))
        print("="*36)

        self.save_checkpoint(0)

        if self.args.model == 'hier':
            self.do_sampling = True
        elif self.args.model == 'cond':
            self.do_sampling = False

        self.actor_critic.train()
        self.agent_pos = np.zeros(
            [self.args.num_steps + 1, self.num_processes_eff, 2], dtype='int')
        # self.visit_count = [np.ones(self.num_processes_eff)]
        self.visit_count = np.ones(
            [self.args.num_steps, self.num_processes_eff], dtype='int')
        self.heuristic_ds = np.zeros(
            [self.args.num_steps, self.num_processes_eff], dtype='int')

        reset_output = self.train_envs.reset()

        obs = reset_output[:, 0]
        info = reset_output[:, 1]
        obs = dict_stack_helper(obs)
        # info = dict_stack_helper(info)
        curr_pos = np.stack([item['agent_pos'] for item in info], 0)
        self.agent_pos[0] = curr_pos
        # [obs] = flatten_batch_dims(obs)
        obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
            for key, obs_i in obs.items()})

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        # time_steps = torch.zeros(self.num_processes_eff, 1).long().to(self.device)
        # episode_rewards = torch.zeros(self.num_processes_eff, 1).to(self.device)
        episode_counter = 0
        episode_rewards = deque(maxlen=300)
        episode_mrids = deque(maxlen=300)
        masks = torch.ones(self.num_processes_eff, 1).float().to(self.device)
        recurrent_hidden_states = torch.zeros(
            self.args.num_steps + 1, self.args.num_processes,
            self.actor_critic.recurrent_hidden_state_size)

        num_updates = int(total_training_steps) // \
            (self.num_processes_eff * self.args.num_steps)
        def batch_iterator(start_idx):
            idx = start_idx
            for _ in range(start_idx, num_updates + self.args.log_interval):
                yield idx
                idx += 1

        start = time.time()

        for iter_id in batch_iterator(start_iter):
            self.actor_critic.train()
            self.rollouts.prev_final_mask.fill_(0)

            for step in range(self.args.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = \
                        self.actor_critic.act(
                            inputs=obs,
                            rnn_hxs=self.rollouts.recurrent_hidden_states[step],
                            masks=self.rollouts.masks[step])

                cpu_actions = action.view(-1).cpu().numpy()

                obs, reward, done, info = self.train_envs.step(cpu_actions)

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(self.device)
                episode_counter += done.sum()

                obs = dict_stack_helper(obs)
                obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
                    for key, obs_i in obs.items()})

                curr_pos = np.stack([item['agent_pos'] for item in info], 0)
                curr_dir = np.stack([item['agent_dir'] for item in info], 0)
                visit_count = np.stack([item['visit_count'] for item in info], 0)

                self.agent_pos[step + 1] = curr_pos
                self.visit_count[step] = visit_count
                if 'is_heuristic_ds' in info[0].keys():
                    is_heuristic_ds = np.stack(
                        [item['is_heuristic_ds'] for item in info], 0)
                    self.heuristic_ds[step] = is_heuristic_ds

                for batch_idx, info_item in enumerate(info):
                    if 'prev_episode' in info_item.keys():
                        prev_final_obs = info_item['prev_episode']['obs']
                        prev_final_obs = DictObs(
                            {key:torch.from_numpy(obs_i).to(self.device) \
                                for key, obs_i in prev_final_obs.items()})
                        self.rollouts.prev_final_mask[step, batch_idx] = 1
                        self.rollouts.prev_final_visit_count[step, batch_idx] = \
                            info_item['visit_count']
                        self.rollouts.prev_final_heur_ds[step, batch_idx] = \
                            float(info_item['is_heuristic_ds'])
                        self.rollouts.prev_final_obs[step, batch_idx].copy_(
                            prev_final_obs)
                        episode_rewards.append(
                            info_item['prev_episode']['info']['episode_reward'])
                        episode_mrids.append(
                            info_item['prev_episode']['info']['max_room_id'])

                reward = torch.from_numpy(reward[:,np.newaxis]).float()
                reward = reward.to(self.device)
                # episode_rewards += reward
                # reward = torch.from_numpy(reward).float()

                # not_done = np.logical_not(done)

                # masks = torch.from_numpy(not_done.astype('float32')).unsqueeze(1)
                # masks = masks.to(self.device)

                self.rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=recurrent_hidden_states,
                    actions=action,
                    action_log_probs=action_log_prob,
                    value_preds=value,
                    rewards=reward,
                    masks=masks,
                )

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    inputs=self.rollouts.obs[-1],
                    rnn_hxs=self.rollouts.recurrent_hidden_states[-1],
                    masks=self.rollouts.masks[-1],
                ).detach()

            anneal_coeff = utils.kl_coefficient_curriculum(
                iter_id=iter_id,
                iters_per_epoch=self.num_batches_per_epoch,
                start_after_epochs=self.args.kl_anneal_start_epochs,
                linear_growth_epochs=self.args.kl_anneal_growth_epochs,
            )

            q_start_flag = utils.q_start_curriculum(
                iter_id=iter_id,
                iters_per_epoch=self.num_batches_per_epoch,
                start_after_epochs=self.args.q_start_epochs,
            )

            if self.args.algo == 'a2c' or self.args.algo == 'acktr':
                # Conditional model
                value_loss, action_loss, dist_entropy,\
                action_log_probs_mean, ic_info, option_info = \
                    self.agent.update(
                        rollouts=self.rollouts,
                        hier_mode=self.args.hier_mode,
                        use_intrinsic_control=False,
                        next_value=next_value,
                        option_space=self.args.option_space,
                        use_ib=self.args.use_infobot,
                        agent_pos=self.agent_pos,
                        bonus_z_encoder=self.z_encoder,
                        b_args=self.b_args,
                        bonus_type=self.args.bonus_type,
                        bonus_normalization=self.args.bonus_normalization,
                        heuristic_ds=self.heuristic_ds,
                        heuristic_coeff=self.args.bonus_heuristic_beta,
                        visit_count=self.visit_count,
                    )

                ic_info.update({
                    'anneal_coeff': anneal_coeff,
                    # 'infobot_coeff': infobot_coeff,
                    'q_start_flag': q_start_flag,
                })

                # if 'traj_ce_loss' in ic_info:
                #     traj_ce_loss.extend(ic_info['traj_ce_loss'])
            else:
                raise ValueError("Unknown algo: {}".format(self.args.algo))

            self.rollouts.after_update()

            total_num_steps = (iter_id + 1) * \
                self.num_processes_eff * self.args.num_steps
            if iter_id % self.args.log_interval == 0:
                if len(episode_rewards) > 1:
                    # cpu_rewards = episode_rewards.cpu().numpy()
                    cpu_rewards = episode_rewards
                    mrids = episode_mrids
                else:
                    cpu_rewards = np.array([0])
                    mrids = np.array([-1])
                end = time.time()
                FPS = int(total_num_steps / (end - start))

                print(f"Updates {iter_id}, num timesteps {total_num_steps}, FPS {FPS}, episodes: {episode_counter} \n Last {len(cpu_rewards)} training episodes: mean/median reward {np.mean(cpu_rewards):.1f}/{np.median(cpu_rewards):.1f}, min/max reward {np.min(cpu_rewards):.1f}/{np.max(cpu_rewards):.1f}")

                print(f" Max room id mean/median: {np.mean(mrids):.1f}/{np.median(mrids):.1f}, min/max: {np.min(mrids)}/{np.max(mrids)}")

                train_success = 1.0 * (np.array(cpu_rewards) > 0)

                self.logger.plot_success(
                    prefix="train_",
                    total_num_steps=total_num_steps,
                    rewards=cpu_rewards,
                    success=train_success,
                    mrids=mrids,
                )
                self.logger.viz.line(total_num_steps, FPS, "FPS", "FPS",
                    xlabel="time_steps")

            if total_num_steps > self.next_val_after:
                print(f"Evaluating success at {total_num_steps} steps")
                self.next_val_after += self.args.val_interval
                val_rewards, val_success, val_mrids = self.eval_success()
                self.logger.plot_success(
                    prefix="val_",
                    total_num_steps=total_num_steps,
                    rewards=val_rewards,
                    success=val_success,
                    mrids=val_mrids,
                    track_best=True,
                )

    def train_infobot_supervised(self, total_training_steps, start_iter):
        """Train loop"""

        print("="*36)
        print("Trainer initialized! Training information:")
        print("\t# of total_training_steps: {}".format(total_training_steps))
        # print("\t# of train envs: {}".format(len(self.train_envs)))
        print("\tnum_processes: {}".format(self.args.num_processes))
        print("\tnum_agents: {}".format(self.args.num_agents))
        # print("\tIterations per epoch: {}".format(self.num_batches_per_epoch))
        print("="*36)

        self.save_checkpoint(0)

        if self.args.model == 'hier':
            self.do_sampling = True
        elif self.args.model == 'cond':
            self.do_sampling = False
        next_save_on = 1 * self.args.save_interval

        self.actor_critic.train()
        # self.agent_pos = np.zeros(
        #     [self.args.num_steps + 1, self.num_processes_eff, 2], dtype='int')
        # self.visit_count = [np.ones(self.num_processes_eff)]
        # self.visit_count = np.ones(
        #     [self.args.num_steps, self.num_processes_eff], dtype='int')
        # self.heuristic_ds = np.zeros(
        #     [self.args.num_steps, self.num_processes_eff], dtype='int')

        reset_output = self.train_envs.reset()

        obs = reset_output[:, 0]
        info = reset_output[:, 1]
        obs = dict_stack_helper(obs)
        # info = dict_stack_helper(info)
        # curr_pos = np.stack([item['agent_pos'] for item in info], 0)
        # self.agent_pos[0] = curr_pos
        # [obs] = flatten_batch_dims(obs)
        obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
            for key, obs_i in obs.items()})

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        # time_steps = torch.zeros(self.num_processes_eff, 1).long().to(self.device)
        # episode_rewards = torch.zeros(self.num_processes_eff, 1).to(self.device)
        episode_counter = 0
        episode_rewards = deque(maxlen=300)
        episode_mrids = deque(maxlen=300)
        ep_len = deque(maxlen=300)

        zz_kld = deque(maxlen=self.args.log_interval)
        zz_kl_loss = deque(maxlen=self.args.log_interval)
        effective_return = deque(maxlen=self.args.log_interval)

        masks = torch.ones(self.num_processes_eff, 1).float().to(self.device)
        recurrent_hidden_states = torch.zeros(
            self.args.num_steps + 1, self.args.num_processes,
            self.actor_critic.recurrent_hidden_state_size)

        num_updates = int(total_training_steps) // \
            (self.num_processes_eff * self.args.num_steps)
        def batch_iterator(start_idx):
            idx = start_idx
            for _ in range(start_idx, num_updates + self.args.log_interval):
                yield idx
                idx += 1

        start = time.time()

        for iter_id in batch_iterator(start_iter):
            self.actor_critic.train()
            self.rollouts.prev_final_mask.fill_(0)

            for step in range(self.args.num_steps):
                with torch.no_grad():
                    z_latent, z_gauss_dist, value, action, \
                    action_log_prob, recurrent_hidden_states = \
                        self.actor_critic.act(
                            inputs=obs,
                            rnn_hxs=self.rollouts.recurrent_hidden_states[step],
                            masks=self.rollouts.masks[step],
                            do_z_sampling=self.args.z_stochastic)
                    z_eps = (z_latent - z_gauss_dist.loc) / z_gauss_dist.scale

                cpu_actions = action.view(-1).cpu().numpy()

                obs, reward, done, info = self.train_envs.step(cpu_actions)

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(self.device)
                episode_counter += done.sum()

                obs = dict_stack_helper(obs)
                obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
                    for key, obs_i in obs.items()})

                # curr_pos = np.stack([item['agent_pos'] for item in info], 0)
                # curr_dir = np.stack([item['agent_dir'] for item in info], 0)
                # visit_count = np.stack([item['visit_count'] for item in info], 0)

                # self.agent_pos[step + 1] = curr_pos
                # self.visit_count[step] = visit_count
                # if 'is_heuristic_ds' in info[0].keys():
                #     is_heuristic_ds = np.stack(
                #         [item['is_heuristic_ds'] for item in info], 0)
                #     self.heuristic_ds[step] = is_heuristic_ds

                for batch_idx, info_item in enumerate(info):
                    if 'prev_episode' in info_item.keys():
                        # prev_final_obs = info_item['prev_episode']['obs']
                        # prev_final_obs = DictObs(
                        #     {key:torch.from_numpy(obs_i).to(self.device) \
                        #         for key, obs_i in prev_final_obs.items()})
                        # self.rollouts.prev_final_mask[step, batch_idx] = 1
                        # self.rollouts.prev_final_visit_count[step, batch_idx] = \
                        #     info_item['visit_count']
                        # self.rollouts.prev_final_heur_ds[step, batch_idx] = \
                        #     float(info_item['is_heuristic_ds'])
                        # self.rollouts.prev_final_obs[step, batch_idx].copy_(
                        #     prev_final_obs)
                        episode_rewards.append(
                            info_item['prev_episode']['info']['episode_reward'])
                        episode_mrids.append(
                            info_item['prev_episode']['info']['max_room_id'])
                        ep_len.append(
                            info_item['prev_episode']['info']['step_count'])

                reward = torch.from_numpy(reward[:,np.newaxis]).float()
                reward = reward.to(self.device)

                self.rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=recurrent_hidden_states,
                    actions=action,
                    action_log_probs=action_log_prob,
                    value_preds=value,
                    z_eps=z_eps,
                    rewards=reward,
                    masks=masks,
                )

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    inputs=self.rollouts.obs[-1],
                    rnn_hxs=self.rollouts.recurrent_hidden_states[-1],
                    masks=self.rollouts.masks[-1],
                ).detach()


            total_num_steps = (iter_id + 1) * \
                self.num_processes_eff * self.args.num_steps

            anneal_coeff = utils.kl_coefficient_curriculum(
                iter_id=total_num_steps,
                iters_per_epoch=1,
                start_after_epochs=self.args.kl_anneal_start_epochs,
                linear_growth_epochs=self.args.kl_anneal_growth_epochs,
            )

            q_start_flag = utils.q_start_curriculum(
                iter_id=total_num_steps,
                iters_per_epoch=1,
                start_after_epochs=self.args.q_start_epochs,
            )

            if not self.args.z_stochastic:
                infobot_coeff = 0
            else:
                infobot_coeff = utils.kl_coefficient_curriculum(
                    iter_id=total_num_steps,
                    iters_per_epoch=1,
                    start_after_epochs=self.args.infobot_kl_start,
                    linear_growth_epochs=self.args.infobot_kl_growth,
                )

            min_ib_coeff = min(self.args.infobot_beta_min, self.args.infobot_beta)
            if self.args.infobot_beta > 0:
                infobot_coeff = max(infobot_coeff, min_ib_coeff / self.args.infobot_beta)
                if not self.args.z_stochastic:
                    infobot_coeff = 0

            if self.args.algo == 'a2c' or self.args.algo == 'acktr':
                # Conditional model
                value_loss, action_loss, dist_entropy,\
                action_log_probs_mean, ic_info = \
                    self.agent.update_infobot_supervised(
                        rollouts=self.rollouts,
                        infobot_beta=self.args.infobot_beta,
                        next_value=next_value,
                        anneal_coeff=infobot_coeff,
                    )

                ic_info.update({
                    'anneal_coeff': infobot_coeff,
                    'q_start_flag': q_start_flag,
                })
                zz_kld.append(ic_info['zz_kld'])
                zz_kl_loss.append(ic_info['zz_kl_loss'])
                effective_return.append(ic_info['effective_return'])

            else:
                raise ValueError("Unknown algo: {}".format(self.args.algo))

            self.rollouts.after_update()


            if iter_id % self.args.log_interval == 0:
                if len(episode_rewards) > 1:
                    # cpu_rewards = episode_rewards.cpu().numpy()
                    cpu_rewards = episode_rewards
                    mrids = episode_mrids
                    episode_length = ep_len
                else:
                    cpu_rewards = np.array([0])
                    mrids = np.array([0])
                    episode_length = np.array([0])
                end = time.time()
                FPS = int(total_num_steps / (end - start))

                print(f"Updates {iter_id}, num timesteps {total_num_steps}, FPS {FPS}, episodes: {episode_counter} \n Last {len(cpu_rewards)} training episodes: mean/median reward {np.mean(cpu_rewards):.1f}/{np.median(cpu_rewards):.1f}, min/max reward {np.min(cpu_rewards):.1f}/{np.max(cpu_rewards):.1f}")

                print(f" Max room id mean/median: {np.mean(mrids):.1f}/{np.median(mrids):.1f}, min/max: {np.min(mrids)}/{np.max(mrids)}")

                train_success = 1.0 * (np.array(cpu_rewards) > 0)

                self.logger.plot_success(
                    prefix="train_",
                    total_num_steps=total_num_steps,
                    rewards=cpu_rewards,
                    success=train_success,
                    mrids=mrids,
                )
                self.logger.viz.line(total_num_steps, FPS, "FPS", "FPS",
                    xlabel="time_steps")
                self.logger.plot_quad_stats(
                    x_val=total_num_steps,
                    array=episode_length,
                    plot_title="episode_length")
                self.logger.viz.line(total_num_steps, np.mean(effective_return),
                    "effective_return", "mean", xlabel="time_steps")
                self.logger.viz.line(total_num_steps, np.mean(zz_kld),
                    "zz_kl", "zz_kld", xlabel="time_steps")
                self.logger.viz.line(total_num_steps, np.mean(zz_kl_loss),
                    "zz_kl", "zz_kl_loss", xlabel="time_steps")
                self.logger.viz.line(total_num_steps, infobot_coeff,
                    "zz_kl", "anneal_coeff", xlabel="time_steps")
                self.logger.viz.line(total_num_steps, np.mean(dist_entropy),
                    "policy_entropy", "entropy", xlabel="time_steps")

            if total_num_steps > self.next_val_after:
                print(f"Evaluating success at {total_num_steps} steps")
                self.next_val_after += self.args.val_interval
                val_rewards, val_success, val_mrids = self.eval_success_td()
                best_success_achieved = self.logger.plot_success(
                    prefix="val_",
                    total_num_steps=total_num_steps,
                    rewards=val_rewards,
                    success=val_success,
                    mrids=val_mrids,
                    track_best=True,
                )
                self.save_checkpoint(total_num_steps, fname="best_val_success.vd")

            if total_num_steps > next_save_on:
                next_save_on += self.args.save_interval
                self.save_checkpoint(total_num_steps)

    def save_checkpoint(self, curr_time_steps, suffix="", fname=None):
        # Save checkpoint
        self.args.curr_time_steps = curr_time_steps
        os.makedirs(self.args.save_dir, exist_ok=True)

        if fname is None:
            fname = self.args.algo + "_td_{:010d}{}.vd".format(
                int(curr_time_steps), suffix)

        save_path = os.path.join(
            self.args.save_dir, fname)

        # A really ugly way to save a model to CPU
        # save_model = actor_critic
        # if args.cuda:
        #     save_model = copy.deepcopy(actor_critic).cpu()

        save_dict = {
          'model': self.actor_critic.state_dict(),
          # 'options_decoder': self.options_decoder.state_dict(),
          'params': vars(self.args),
          # 'policy_kwargs' : policy_kwargs,
          # 'train_kwargs' : train_kwargs
          'args_state': self.args_state,
        }

        save_dict['optimizer'] = \
            self.agent.optimizer.state_dict()

        print("Currently on visdom env: {}".format(self.args.visdom_env_name))
        print("Saving checkpoint:", save_path)
        torch.save(save_dict, save_path)

    def eval_success(self):
        with torch.no_grad():
            self.val_envs.reset_config_rng()
            assert self.val_envs.get_attr('reset_on_done')[0]
            self.actor_critic.train()
            reset_output = self.val_envs.reset()

            obs = reset_output[:, 0]
            info = reset_output[:, 1]
            obs = dict_stack_helper(obs)
            obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
                for key, obs_i in obs.items()})

            episode_counter = 0
            episode_rewards = np.zeros((self.args.num_eval_episodes))
            episode_mrids = np.zeros((self.args.num_eval_episodes))
            masks = torch.ones(self.num_processes_eff, 1).float().to(self.device)
            recurrent_hidden_states = torch.zeros(self.args.num_processes,
                self.actor_critic.recurrent_hidden_state_size).to(self.device)

            eval_done = False
            while not eval_done:
                value, action, action_log_prob, recurrent_hidden_states = \
                    self.actor_critic.act(
                        inputs=obs,
                        rnn_hxs=recurrent_hidden_states,
                        masks=masks)

                cpu_actions = action.view(-1).cpu().numpy()
                obs, _, done, info = self.val_envs.step(cpu_actions)
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(self.device)

                obs = dict_stack_helper(obs)
                obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
                    for key, obs_i in obs.items()})

                for batch_idx, info_item in enumerate(info):
                    if 'prev_episode' in info_item.keys():
                        episode_rewards[episode_counter] = \
                            info_item['prev_episode']['info']['episode_reward']
                        episode_mrids[episode_counter] = \
                            info_item['prev_episode']['info']['max_room_id']

                        episode_counter += 1
                        if episode_counter >= self.args.num_eval_episodes:
                            eval_done = True
                            break

            episode_success = 1.0 * (episode_rewards > 0)
            return episode_rewards, episode_success, episode_mrids

    def eval_success_td(self):
        with torch.no_grad():
            self.val_envs.reset_config_rng()
            assert self.val_envs.get_attr('reset_on_done')[0]
            self.actor_critic.train()
            reset_output = self.val_envs.reset()

            obs = reset_output[:, 0]
            info = reset_output[:, 1]
            obs = dict_stack_helper(obs)
            obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
                for key, obs_i in obs.items()})

            episode_counter = 0
            episode_rewards = np.zeros((self.args.num_eval_episodes))
            episode_mrids = np.zeros((self.args.num_eval_episodes))
            masks = torch.ones(self.num_processes_eff, 1).float().to(self.device)
            recurrent_hidden_states = torch.zeros(self.args.num_processes,
                self.actor_critic.recurrent_hidden_state_size).to(self.device)

            eval_done = False
            while not eval_done:
                z_latent, z_gauss_dist, value, action, \
                action_log_prob, recurrent_hidden_states = \
                    self.actor_critic.act(
                        inputs=obs,
                        rnn_hxs=recurrent_hidden_states,
                        masks=masks,
                        do_z_sampling=False)

                cpu_actions = action.view(-1).cpu().numpy()
                obs, _, done, info = self.val_envs.step(cpu_actions)
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(self.device)

                obs = dict_stack_helper(obs)
                obs = DictObs({key:torch.from_numpy(obs_i).to(self.device) \
                    for key, obs_i in obs.items()})

                for batch_idx, info_item in enumerate(info):
                    if 'prev_episode' in info_item.keys():
                        episode_rewards[episode_counter] = \
                            info_item['prev_episode']['info']['episode_reward']
                        episode_mrids[episode_counter] = \
                            info_item['prev_episode']['info']['max_room_id']

                        episode_counter += 1
                        if episode_counter >= self.args.num_eval_episodes:
                            eval_done = True
                            break

            episode_success = 1.0 * (episode_rewards > 0)
            return episode_rewards, episode_success, episode_mrids
