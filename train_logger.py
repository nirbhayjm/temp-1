from typing import Dict, Optional, Tuple

import time
import numpy as np
import scipy
import os
import pprint

from collections import namedtuple

import torch
from distributions import FixedCategorical
import torch.distributions as ds
import torchvision
import utilities.utilities as utils
from utilities.utilities import (MovingAverage,
                                 MovingAverageMoment,
                                 MovingAverageHeatMap,
                                 DictArrayQueue)

from utilities.visualize import VisdomLogger


class TrainLogger(object):
    def __init__(self,
        args,
        vis_env,
        val_envs,
        device: torch.device,
        num_batches_per_epoch: int,
        num_processes_eff: int,
        continuous_state_space: bool,
    ):
        self.args = args
        self.vis_env = vis_env
        self.val_envs = val_envs
        self.device = device
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_processes_eff = num_processes_eff
        self.continuous_state_space = continuous_state_space

        if not self.continuous_state_space:
            if hasattr(self.vis_env.actions, 'forward'):
                self.action_space_type = 'pov'
                # assert hasattr(self.vis_env, 'dir_to_cardinal')
                # self.dir_to_cardinal = self.vis_env.dir_to_cardinal
            elif hasattr(self.vis_env.actions, 'up'):
                self.action_space_type = 'cardinal'
            else:
                raise ValueError

        # self.vis_env.gen_occupancy_grid()
        # self.vis_env.reset()
        self.init_visdom_env()
        self.init_moving_averages()
        self.init_scalars()
        self.init_tensors()
        if self.args.vis_heatmap:
            self.init_heatmap_logging()

    def init_visdom_env(self):
        self.args.output_log_path = os.path.join(
            self.args.log_dir, self.args.time_id + '_' + self.args.visdom_env_name + '.log')
        self.viz = VisdomLogger(env_name=self.args.visdom_env_name,
                                server=self.args.server,
                                port=self.args.port,
                                log_file=self.args.output_log_path,
                                fig_save_dir=self.args.save_dir,
                                win_prefix=self.args.identifier)
                                # win_prefix=self.args.time_id + '_' + self.args.identifier)
        print("Logging visdom update events to: {}".format(
            self.args.output_log_path))
        pprint.pprint(vars(self.args))
        # self.viz.text(pprint.pformat(vars(self.args), indent=4), "params")
        self.viz.text(vars(self.args), "params")

        vis_obs, vis_info = self.vis_env.reset()
        assert 'rgb_grid' in vis_info
        self.env_rgb_img = vis_info['rgb_grid'].transpose([2, 0, 1])
        self.env_rgb_img = np.flip(self.env_rgb_img, 1)
        self.viz.image(self.env_rgb_img, 'env_rgb_img')

    def init_moving_averages(self):
        M_AVG_WIN_SIZE = 1000
        queue_size = max(100, 5000 // self.num_processes_eff)
        self.m_traj_enc_ll = MovingAverage(self.args.omega_curr_win_size)
        self.m_reward = MovingAverageMoment(self.args.log_interval, use_median=True)
        self.m_eff_return = MovingAverage(M_AVG_WIN_SIZE)
        self.m_dist_entropy = MovingAverage(M_AVG_WIN_SIZE)
        self.m_value_loss = MovingAverage(M_AVG_WIN_SIZE)
        self.m_value_pred = MovingAverage(M_AVG_WIN_SIZE)
        self.m_bonus_reward = MovingAverage(M_AVG_WIN_SIZE)
        self.m_bonus_std = MovingAverage(10)
        self.m_empowerment = MovingAverage(M_AVG_WIN_SIZE)
        self.m_empowerment_sum_t = MovingAverage(M_AVG_WIN_SIZE)
        self.m_action_loss = MovingAverage(M_AVG_WIN_SIZE)
        # m_pathd_loss = MovingAverage(M_AVG_WIN_SIZE)
        self.m_option_loss = MovingAverage(M_AVG_WIN_SIZE)
        self.m_opt_value_loss = MovingAverage(M_AVG_WIN_SIZE)
        self.m_opt_returns = MovingAverage(M_AVG_WIN_SIZE)

        self.m_episode_len = MovingAverage(M_AVG_WIN_SIZE)
        self.m_success = MovingAverage(M_AVG_WIN_SIZE)
        self.m_success_0 = MovingAverage(M_AVG_WIN_SIZE)
        self.m_success_1 = MovingAverage(M_AVG_WIN_SIZE)
        self.m_alp_mean = MovingAverage(M_AVG_WIN_SIZE)
        # self.m_alp_tensor = DictArrayQueue(
        #     queue_size,
        #     shape_dict={'action_log_probs': (num_steps, num_processes, 1)}
        # )
        # moving_avg_reward = 0.0
        self.MOVING_AVG_DECAY = 1.0 - 1e-3

        # if self.args.use_infobot:
        num_steps = self.args.num_steps
        z_latent_dims = self.args.z_latent_dims
        num_processes = self.num_processes_eff
        action_dim = self.vis_env.action_space.n
        if not self.continuous_state_space:
            self.m_opt_track_q = DictArrayQueue(
                queue_size,
                shape_dict={
                    's_f_pos': (num_processes, 2),
                    'agent_pos': (num_steps + 1, num_processes, 2),
                    'agent_dir': (num_steps + 1, num_processes, 1),
                    'masks': (num_steps + 1, num_processes, 1),
                    'omega': (num_processes, self.args.omega_option_dims),
                    'ep_len': (num_processes,),
                    'zz_kl_tensor': (num_steps + 1, num_processes, 1),
                    'pi_kl_tensor': (num_steps, num_processes, 1),
                    # 'z_loc': (num_steps + 1, num_processes, z_latent_dims),
                    # 'z_std': (num_steps + 1, num_processes, z_latent_dims),
                    # 'action_log_probs': (num_steps, num_processes, 1),
                    'action_probs': (num_steps, num_processes, action_dim),
                    'default_action_probs': (num_steps, num_processes, action_dim),
                }
            )
        else:
            queue_size = min(3, 5000 // self.num_processes_eff)
            self.m_opt_track_q = DictArrayQueue(
                queue_size,
                shape_dict={
                    's_f_pos_velocity': (num_processes, 2),
                    'pos_velocity': (num_steps + 1, num_processes, 2),
                    'masks': (num_steps + 1, num_processes, 1),
                    'omega': (num_processes, self.args.omega_option_dims),
                    'ep_len': (num_processes,),
                    'zz_kl_tensor': (num_steps + 1, num_processes, 1),
                    'pi_kl_tensor': (num_steps, num_processes, 1),
                    'action_probs': (num_steps, num_processes, action_dim),
                    'default_action_probs': (num_steps, num_processes, action_dim),
                }
            )

    def init_scalars(self):
        pass

    def init_tensors(self):
        self.episode_rewards = torch.zeros([self.num_processes_eff, 1])
        self.final_rewards = torch.zeros([self.num_processes_eff, 1])
        if not self.continuous_state_space:
            self.agent_pos = np.zeros(
                [self.args.num_steps + 1, self.num_processes_eff, 2], dtype='int')
            self.agent_dir = np.zeros(
                [self.args.num_steps + 1, self.num_processes_eff, 1], dtype='int')
        else:
            self.pos_velocity = np.zeros(
                [self.args.num_steps + 1, self.num_processes_eff, 2], dtype='float32')
        self.step_counter = np.zeros((self.num_processes_eff))

    def init_heatmap_logging(self):
        if self.continuous_state_space:
            return

        render_shape = (self.vis_env.width, self.vis_env.height)
        self.m_istate_hmap = MovingAverageHeatMap(render_shape, self.args.log_interval)
        self.m_fstate_hmap = MovingAverageHeatMap(render_shape, self.args.log_interval)
        self.m_all_steps_hmap = MovingAverageHeatMap(render_shape, self.args.log_interval)

        if self.args.hier_mode == 'bonus':
            self.m_bonus_c = MovingAverageHeatMap(render_shape, self.args.log_interval)
            self.m_bonus_kl = MovingAverageHeatMap(render_shape, self.args.log_interval)

        # self.m_alp_tensor = MovingAverageHeatMap(
        #     (num_steps, num_processes, 1), self.args.log_interval)
        action_dims = self.vis_env.action_space.n
        if self.action_space_type == 'cardinal':
            pi_shape = (action_dims, *render_shape)
        elif self.action_space_type == 'pov':
            pi_shape = (4 + 4, *render_shape)
        self.m_pi_def = MovingAverageHeatMap(pi_shape, self.args.log_interval)
        self.m_pi_def_kl = MovingAverageHeatMap(render_shape, self.args.log_interval,
            freq_normalize=True)

        if self.args.option_space == 'discrete':
            self.m_option_choice = MovingAverageHeatMap(
                (self.args.omega_option_dims,), self.args.log_interval)

            self.m_option_g1 = MovingAverageHeatMap(
                (self.args.omega_option_dims,), self.args.log_interval)
            self.m_option_g2 = MovingAverageHeatMap(
                (self.args.omega_option_dims,), self.args.log_interval)

            self.m_options = {
                'all_states': [MovingAverageHeatMap(render_shape, \
                    self.args.log_interval) for _ in range(self.args.omega_option_dims)],
                'final_state': [MovingAverageHeatMap(render_shape, \
                    self.args.log_interval) for _ in range(self.args.omega_option_dims)],
                # 'pi_def': [MovingAverageHeatMap(
                #     pi_shape, self.args.log_interval, freq_normalize=True) \
                #         for _ in range(self.args.omega_option_dims)],
                'pi_def_kl': [MovingAverageHeatMap(
                    render_shape, self.args.log_interval, freq_normalize=True) \
                        for _ in range(self.args.omega_option_dims)],
            }

    def reset_tensors(self):
        # Setup for this update iteration
        self.episode_rewards.fill_(0)
        self.final_rewards.fill_(0)
        self.step_counter *= 0

    def update_at_step(
        self,
        step,
        reward_t,
        not_done,
        agent_dir=None,
        agent_pos=None,
        pos_velocity=None,
    ):
        if not self.continuous_state_space:
            self.agent_pos[step + 1] = agent_pos
            self.agent_dir[step + 1] = agent_dir[:, np.newaxis]
        else:
            self.pos_velocity[step + 1] = pos_velocity

        self.step_counter += not_done
        # self.episode_rewards += reward_t * \
        #     torch.from_numpy(not_done.astype('float32')).unsqueeze(-1)
        self.episode_rewards += reward_t

    def on_train_start(self):
        self.start_t = time.time()

    def on_iter_start(self, info):
        self.reset_tensors()
        if not self.continuous_state_space:
            self.agent_pos[0] = info['agent_pos']
            self.agent_dir[0] = info['agent_dir'][:, np.newaxis]
        else:
            self.pos_velocity[0] = info['pos_velocity']

    def on_iter_end(
        self,
        start_iter,
        iter_id,
        total_time_steps,
        rollouts,
        omega_option,
        omega_dim_current,
        omega_dim_ll_threshold,
        value_loss,
        action_loss,
        dist_entropy,
        action_log_probs_mean,
        ic_info,
        option_info,
        eval_info,
    ):

        option_loss = option_info['option_loss']
        effective_return = ic_info['effective_return']
        empowerment_value = ic_info['empowerment_value']
        if 'empowerment_sum_t' in ic_info:
            empowerment_sum_t = ic_info['empowerment_sum_t']
        traj_enc_ll = ic_info['p_ll']
        action_dist_probs = ic_info['action_dist_probs']

        batch_weights = np.squeeze(ic_info['batch_weights'])
        batch_weights /= batch_weights.sum()
        batch_weights_entropy = -(batch_weights * np.log(1e-7 + batch_weights)).sum()
        ic_info['batch_weights_entropy'] = batch_weights_entropy

        with torch.no_grad():
            cpu_episode_rewards = self.episode_rewards.cpu().numpy()
            return_mean = cpu_episode_rewards.mean()
            ep_len_mean = (self.step_counter.astype('int') + 1).mean()
            # success_mean = success_train.mean()
            # value_pred = rollouts.value_preds * rollouts.masks
            # value_pred = value_pred.cpu().numpy().sum(0).mean()
            value_pred = rollouts.value_preds[0]
            value_pred = value_pred.cpu().numpy().mean()

            self.m_reward.add(cpu_episode_rewards)
            if 'bonus_reward' in ic_info:
                self.m_bonus_reward.add(ic_info['bonus_reward'])
                np_mask = rollouts.masks.cpu().numpy()
                self.m_bonus_std.add(
                    ic_info['bonus_tensor'][np_mask.astype('bool')].std())
            self.m_eff_return.add(effective_return)
            self.m_value_pred.add(value_pred)
            self.m_empowerment.add(empowerment_value)
            if 'empowerment_sum_t' in ic_info:
                self.m_empowerment_sum_t.add(empowerment_sum_t)
            self.m_traj_enc_ll.add(traj_enc_ll)
            self.m_action_loss.add(action_loss)
            # m_pathd_loss.add(path_derivative_loss)
            self.m_option_loss.add(option_loss)
            self.m_alp_mean.add(action_log_probs_mean)
            # self.m_alp_tensor.add({'action_log_probs': action_log_probs})
            self.m_value_loss.add(value_loss)
            self.m_episode_len.add(ep_len_mean)
            self.m_dist_entropy.add(dist_entropy)

            # if self.args.hier_mode == 'bonus':
            #     import pdb; pdb.set_trace()
            #     bonus_c_tensor = ic_info['bonus_c_tensor']
            #     self.m_bonus_c.add()
            #     # self.m_bonus_kl

            if self.args.hier_mode == 'transfer':
                self.m_opt_value_loss.add(option_info['option_value_loss'])
                self.m_opt_returns.add(option_info['option_returns'])

            # if self.args.hier_mode == 'transfer':
            #     omega_option = np.zeros(
            #         (self.num_processes_eff, self.args.omega_option_dims))
            # else:
            #     # omega_option = omega_option.cpu().numpy()
            #     pass

            if not self.continuous_state_space:
                track_dict = {
                    's_f_pos': self.agent_pos[-1],
                    'agent_pos': self.agent_pos,
                    'agent_dir': self.agent_dir,
                    'masks': rollouts.masks.cpu().numpy(),
                    # 'zz_kl_tensor': ic_info['zz_kl_tensor'],
                    'omega': omega_option,
                    'ep_len': self.step_counter + 1,
                }

                track_dict['action_probs'] = ic_info['action_probs']
                if self.args.use_infobot:
                    # track_dict['pi_kl_tensor'] = ic_info['pi_kl_tensor']
                    # track_dict['z_loc'] = ic_info['z_loc']
                    # track_dict['z_std'] = ic_info['z_std']
                    track_dict['zz_kl_tensor'] = ic_info['zz_kl_tensor']
                    track_dict['default_action_probs'] = ic_info['default_action_probs']
                    # track_dict['action_log_probs'] = ic_info['action_log_probs']

                self.m_opt_track_q.add(track_dict)

                if self.args.vis_heatmap:
                    self.update_heatmaps(
                        rollouts=rollouts,
                        omega_option=omega_option,
                        omega_dim_current=omega_dim_current,
                        action_dist_probs=action_dist_probs,
                        # ic_info=ic_info,
                    )
            else:
                track_dict = {
                    's_f_pos_velocity': self.pos_velocity[-1],
                    'pos_velocity': self.pos_velocity,
                    'masks': rollouts.masks.cpu().numpy(),
                    'omega': omega_option,
                    'ep_len': self.step_counter + 1,
                }

                track_dict['action_probs'] = ic_info['action_probs']
                if self.args.use_infobot:
                    track_dict['zz_kl_tensor'] = ic_info['zz_kl_tensor']
                    track_dict['default_action_probs'] = ic_info['default_action_probs']

                self.m_opt_track_q.add(track_dict)

                # if self.args.vis_heatmap:
                #     self.update_continuous_heatmaps(
                #         rollouts=rollouts,
                #         omega_option=omega_option,
                #         omega_dim_current=omega_dim_current,
                #         action_dist_probs=action_dist_probs,
                #         # ic_info=ic_info,
                #     )

        if iter_id % self.args.log_interval == 0:
            self.end_t = time.time()
            total_num_steps = (iter_id - start_iter + 1) * \
                self.num_processes_eff * self.args.num_steps
            print("Updates {}, num timesteps {}, epochs {:.2f}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
               format(iter_id, total_num_steps,
                      iter_id / self.num_batches_per_epoch,
                      int(total_num_steps / (self.end_t - self.start_t)),
                      cpu_episode_rewards.mean(),
                      np.median(cpu_episode_rewards),
                      cpu_episode_rewards.min(),
                      cpu_episode_rewards.max(), dist_entropy,
                      value_loss, action_loss))

            # if 'ic_adaptive_wts' in ic_info:
            #     print("IC:", ic_info['ic_adaptive_wts'])

            if self.args.vis:
                self.update_visdom_plots(
                    start_iter=start_iter,
                    iter_id=iter_id,
                    total_time_steps=total_time_steps,
                    return_mean=return_mean,
                    dist_entropy=dist_entropy,
                    cpu_episode_rewards=cpu_episode_rewards,
                    omega_dim_current=omega_dim_current,
                    omega_dim_ll_threshold=omega_dim_ll_threshold,
                    ic_info=ic_info,
                )

        if not self.continuous_state_space \
        and iter_id % (self.args.heatmap_interval) == 0:
            if self.args.vis_heatmap:
                self.update_visdom_heatmaps(
                    iter_id=iter_id,
                    omega_option=omega_option,
                    omega_dim_current=omega_dim_current,
                    rollouts=rollouts,
                    ic_info=ic_info,
                )

        if not self.continuous_state_space \
        and iter_id % (self.args.heatmap_interval) == 0 \
        and self.args.model != 'cond' \
        and self.args.option_space == 'discrete' \
        and self.args.hier_mode != 'infobot-supervised' \
        and self.args.env_name != 'crossing':
            # # if self.args.use_infobot:
            # if iter_id % (self.args.log_interval * 10) == 0:
            # if iter_id % (self.args.log_interval * 10) == 0:
            # print("[Eval] Evaluating default behavior for IB policy...")
            # assert self.args.hier_mode == 'vic'
            info_on_policy = self.update_on_policy_averages(iter_id, omega_dim_current)

            # if 'kl_zz_grid' in eval_info:
            if eval_info:
                info_off_policy = self.update_off_policy_plots(iter_id, eval_info)
                info_on_policy.update(info_off_policy)

            self.plot_grouped_heatmaps(iter_id, info_on_policy)

        if self.continuous_state_space \
        and iter_id % (self.args.heatmap_interval) == 0 \
        and self.args.env_name == 'mountain-car' \
        and self.args.model != 'cond' \
        and self.args.hier_mode != 'infobot-supervised':
            info_on_policy = self.update_on_policy_continuous(
                iter_id, omega_dim_current)

        # Saving visdom env to disk
        if iter_id % (self.args.log_interval) == 0 \
        and iter_id > 0:
            # print("Saving visdom env to disk: {}".format(self.args.visdom_env_name))
            self.viz.viz.save([self.args.visdom_env_name])

        if iter_id % (self.args.log_interval) == 0:
            win_data = self.viz.viz.get_window_data(
                env=self.args.visdom_env_name, win=None)
            save_path = os.path.join(self.args.save_dir, "visdom_windows.json")
            # print("Saving visdom windows to file: {}".format(save_path))
            with open(save_path, 'w') as f_output:
                f_output.write(win_data)

        return self.m_traj_enc_ll.value()[0], self.m_empowerment.value()[0]

    def plot_grouped_heatmaps(self, iter_id, info):
        # 'off_policy_opt_pi_kl'
        # 'off_policy_opt_z_kl'
        # 'on_policy_opt_z_kl'
        # 'on_policy_opt_pi_kl'
        pi_kl_plots = []
        pi_kl_subtitles = []
        z_kl_plots = []
        z_kl_subtitles = []
        for key, val in info.items():
            if 'pi_kl' in key:
                nrow = val['plot'].shape[0]
                pi_kl_plots.append(val['plot'])
                pi_kl_subtitles.extend(val['subtitles'])
            elif 'z_kl' in key:
                z_kl_plots.append(val['plot'])
                z_kl_subtitles.extend(val['subtitles'])
            else:
                raise ValueError

        if pi_kl_plots:
            pi_kl_plots = np.concatenate(pi_kl_plots, 0)
            self.viz.plotly_grid('heatmap',
                pi_kl_plots,
                subplot_titles=pi_kl_subtitles,
                ncols=nrow,
                key="pi_kl_opt",
                normalize=False,
                iter_id=iter_id,
                save_figures=True,
            )

        if z_kl_plots:
            z_kl_plots = np.concatenate(z_kl_plots, 0)
            self.viz.plotly_grid('heatmap',
                z_kl_plots,
                subplot_titles=z_kl_subtitles,
                ncols=nrow,
                key="z_kl_opt",
                normalize=False,
                iter_id=iter_id,
                save_figures=True,
            )

    def update_on_policy_averages(self, iter_id, omega_dim_current):
        info = {}
        op_q = self.m_opt_track_q.get_all_items()

        op_q['s_f_pos'] = op_q['s_f_pos'].reshape(
            -1, *op_q['s_f_pos'].shape[2:])
        assert op_q['s_f_pos'].min() >= 0

        op_q['omega'] = op_q['omega'].reshape(
            -1, *op_q['omega'].shape[2:])

        op_q['ep_len'] = op_q['ep_len'].reshape(
            -1, *op_q['ep_len'].shape[2:])

        op_q['agent_pos'] = op_q['agent_pos'].transpose([0, 2, 1, 3])
        op_q['agent_pos'] = op_q['agent_pos'].reshape(
            -1, *op_q['agent_pos'].shape[2:])

        op_q['agent_dir'] = op_q['agent_dir'].transpose([0, 2, 1, 3])
        op_q['agent_dir'] = op_q['agent_dir'].reshape(
            -1, *op_q['agent_dir'].shape[2:])

        op_q['masks'] = op_q['masks'].transpose([0, 2, 1, 3])
        op_q['masks'] = op_q['masks'].reshape(
            -1, *op_q['masks'].shape[2:])

        # op_q['z_loc'] = op_q['z_loc'].transpose([0, 2, 1, 3])
        # op_q['z_loc'] = op_q['z_loc'].reshape(
        #     -1, *op_q['z_loc'].shape[2:])
        #
        # op_q['z_std'] = op_q['z_std'].transpose([0, 2, 1, 3])
        # op_q['z_std'] = op_q['z_std'].reshape(
        #     -1, *op_q['z_std'].shape[2:])

        op_q['zz_kl_tensor'] = op_q['zz_kl_tensor'].transpose([0, 2, 1, 3])
        op_q['zz_kl_tensor'] = op_q['zz_kl_tensor'].reshape(
            -1, *op_q['zz_kl_tensor'].shape[2:])

        op_q['pi_kl_tensor'] = op_q['pi_kl_tensor'].transpose([0, 2, 1, 3])
        op_q['pi_kl_tensor'] = op_q['pi_kl_tensor'].reshape(
            -1, *op_q['pi_kl_tensor'].shape[2:])

        op_q['action_probs'] = \
            op_q['action_probs'].transpose([0, 2, 1, 3])
        op_q['action_probs'] = \
            op_q['action_probs'].reshape(
                -1, *op_q['action_probs'].shape[2:])

        op_q['default_action_probs'] = \
            op_q['default_action_probs'].transpose([0, 2, 1, 3])
        op_q['default_action_probs'] = \
            op_q['default_action_probs'].reshape(
                -1, *op_q['default_action_probs'].shape[2:])

        # op_q['action_log_probs'] = \
        #     op_q['action_log_probs'].transpose([0, 2, 1, 3])
        # op_q['action_log_probs'] = \
        #     op_q['action_log_probs'].reshape(
        #         -1, *op_q['action_log_probs'].shape[2:])

        op_q['s_f_pos'] = op_q['s_f_pos'].astype('int')
        op_q['ep_len'] = op_q['ep_len'].astype('int')
        op_q['agent_pos'] = op_q['agent_pos'].astype('int')
        op_q['agent_dir'] = op_q['agent_dir'].astype('int')
        n_items = op_q['s_f_pos'].shape[0]

        GROUP_BY = 's_f'
        if self.args.option_space == 'discrete':
            GROUP_BY = 'option'
        else:
            GROUP_BY = 's_f'
        assert GROUP_BY in ['s_f', 'option']

        if GROUP_BY == 's_f':
            assert op_q['s_f_pos'].shape == (n_items, 2)
            XYTuple = namedtuple("XYTuple", ["x", "y"])
            s_f_dict = {}
            for idx in range(n_items):
                ep_len = op_q['ep_len'][idx]
                if ep_len < (self.args.num_steps // 2):
                    continue
                s_f_pos = op_q['s_f_pos'][idx]
                s_f = XYTuple(x = s_f_pos[0], y = s_f_pos[1])
                # for key, item in op_q.items():
                #     if idx >= item.shape[0]:
                #         import pdb; pdb.set_trace()
                #         print("WHAT?")
                qu_item = {key: item[idx] for key, item in op_q.items()}
                if s_f in s_f_dict.keys():
                    s_f_dict[s_f].append(qu_item)
                else:
                    s_f_dict[s_f] = [qu_item]

            len_tuples = []
            for key in s_f_dict:
                len_tuples.append((key, len(s_f_dict[key])))

            len_tuples = sorted(len_tuples, key=lambda x:x[1], reverse=True)
            selected_keys = len_tuples
            # TRAJ_DISPLAY_LIMIT = 25
            # selected_keys = len_tuples[:TRAJ_DISPLAY_LIMIT]
        else:
            opt_dict = {key:[] for key in range(omega_dim_current)}
            for idx in range(n_items):
                _opt_id = op_q['omega'][idx].argmax()
                qu_item = {key: item[idx] for key, item in op_q.items()}
                opt_dict[_opt_id].append(qu_item)

            # for opt_id in opt_dict.keys():
            #     for key in opt_dict[opt_id][0].keys():
            #         item = [opt_dict[opt_id][j][key] \
            #             for j in range(len(opt_dict[opt_id]))]
            #         opt_dict[key] = np.stack(item, 0)

        def pov_to_cardinal(action_probs, dir):
            ap_cardinal = np.zeros((action_probs.shape[0], 4 + 6))
            fwd_prob = action_probs[:, self.vis_env.actions.forward]
            left_prob = action_probs[:, self.vis_env.actions.left]
            right_prob = action_probs[:, self.vis_env.actions.right]
            # _other_prob = action_probs[:,
            #     np.arange(action_probs.shape[1]) != self.vis_env.actions.forward]
            ap_cardinal[np.arange(action_probs.shape[0]),
                dir[:, 0]] = fwd_prob
            ap_cardinal[:, 4] = left_prob
            ap_cardinal[:, 5] = right_prob
            ap_cardinal[:, 6:] = action_probs[:, 3:]
            # _dap_map = self.get_hmap(_pos, ap_cardinal, freq_normalize=True)
            return ap_cardinal

        if GROUP_BY == 's_f':
            raise NotImplementedError
            s_titles, kl_titles = [], []
            dap_titles, ap_titles = [], []
            s_maps, avg_s = [], []
            kl_maps, avg_kl = [], []
            dap_maps, avg_dap = [], []
            ap_maps, avg_ap = [], []
            if self.args.option_space == 'discrete':
                opt_pos_maps = [[] for _ in range(omega_dim_current)]
                opt_z_kl_maps = [[] for _ in range(omega_dim_current)]
                opt_pi_kl_maps = [[] for _ in range(omega_dim_current)]
                opt_ap = [[] for _ in range(omega_dim_current)]


            for s_f, hit_freq in selected_keys:
                s_map = None
                _all_pos = []
                # _all_dir = []
                _all_kl = []
                _all_dap = []
                _all_ap = []
                for tau in s_f_dict[s_f]:
                    # _pos = tau['agent_pos'][:tau['ep_len'] + 1]
                    _opt = tau['omega']
                    _pos = tau['agent_pos'][:tau['ep_len']]
                    _dir = tau['agent_dir'][:tau['ep_len']]
                    # _uniq_pos = np.vstack({tuple(row) for row in _pos})
                    _pos_map = self.get_hmap(_pos)

                    if self.args.option_space == 'discrete':
                        _pi_kl = tau['pi_kl_tensor'][:tau['ep_len']]
                        _pi_kl_map = self.get_hmap(_pos, _pi_kl, freq_normalize=True)
                        opt_pos_maps[_opt.argmax()].append(_pos_map)
                        opt_pi_kl_maps[_opt.argmax()].append(_pi_kl_map)

                    # _all_dir.append(_dir)
                    if self.args.use_infobot:
                        # _kl = tau['zz_kl_tensor'][:tau['ep_len'] + 1]
                        _kl = tau['zz_kl_tensor'][:tau['ep_len']]
                        _kl_map = self.get_hmap(_pos, _kl, freq_normalize=True)
                        _dap = tau['default_action_probs'][:tau['ep_len']]
                        _ap = tau['action_probs'][:tau['ep_len']]
                        if self.action_space_type == 'pov':
                            # _dap_cardinal = np.zeros((_dap.shape[0], 4 + 4))
                            # _fwd_prob = _dap[:, self.vis_env.actions.forward]
                            # # _other_prob = _dap[:,
                            # #     np.arange(_dap.shape[1]) != self.vis_env.actions.forward]
                            # _dap_cardinal[np.arange(_dap.shape[0]),
                            #     _dir[:, 0]] = _fwd_prob
                            # _dap_cardinal[:, 4:] = _dap[:, 3:]
                            _dap_cardinal = pov_to_cardinal(_dap, _dir)
                            _ap_cardinal = pov_to_cardinal(_ap, _dir)
                            _dap_map = self.get_hmap(_pos, _dap_cardinal, freq_normalize=True)
                            _ap_map = self.get_hmap(_pos, _ap_cardinal, freq_normalize=True)
                        else:
                            _dap_map = self.get_hmap(_pos, _dap, freq_normalize=True)
                            _ap_map = self.get_hmap(_pos, _ap, freq_normalize=True)
                            # _all_dap.append(_dap)
                        _all_dap.append(_dap_map)
                        _all_ap.append(_ap_map)
                        _all_kl.append(_kl_map)

                        if self.args.option_space == 'discrete':
                            opt_z_kl_maps[_opt.argmax()].append(_kl_map)
                            opt_ap[_opt.argmax()].append(_ap_map)

                    _all_pos.append(_pos_map)

                # _all_pos = np.concatenate(_all_pos, 0)
                _all_pos = np.stack(_all_pos, 0)
                # _all_dir = np.concatenate(_all_dir, 0)
                avg_s.append(_all_pos)
                # avg_d.append(_all_dir)
                # s_map = self.get_hmap(_all_pos)
                s_map = utils.normalize_map(_all_pos)
                # s_maps.append(s_map)
                s_maps.append(s_map)
                str_s_f = "({}, {}) c{}".format(s_f.x, s_f.y, hit_freq)
                s_titles.append(str_s_f)

                if self.args.use_infobot:
                    # _all_kl = np.concatenate(_all_kl, 0)
                    # _all_dap = np.concatenate(_all_dap, 0)
                    _all_kl = np.stack(_all_kl, 0)
                    _all_dap = np.stack(_all_dap, 0)
                    _all_ap = np.stack(_all_ap, 0)

                    avg_kl.append(_all_kl)
                    avg_dap.append(_all_dap)
                    avg_ap.append(_all_ap)
                    assert _all_pos.shape[0] == _all_kl.shape[0]
                    # kl_map = self.get_hmap(_all_pos, _all_kl, freq_normalize=True)
                    # dap_map = self.get_hmap(_all_pos, _all_dap, freq_normalize=True)
                    kl_map = utils.normalize_map(_all_kl)
                    dap_map = utils.normalize_map(_all_dap)
                    ap_map = utils.normalize_map(_all_ap)
                    kl_maps.append(kl_map)
                    dap_maps.append(dap_map)
                    ap_maps.append(ap_map)
                    kl_titles.append("KL" + str_s_f)
                    dap_titles.append("DAP" + str_s_f)
                    ap_titles.append("AP" + str_s_f)
        else:
            _H = self.vis_env.height
            _W = self.vis_env.width
            make_op_list = lambda size: [np.zeros(size) \
                for _ in range(omega_dim_current)]
            opt_pos_maps = make_op_list((_H, _W))
            opt_z_kl_maps = make_op_list((_H, _W))
            # opt_pi_kl_maps = make_op_list()
            if self.action_space_type == 'pov':
                opt_ap = make_op_list((4 + 6, _H, _W))
            else:
                opt_ap = make_op_list(
                    (self.vis_env.action_space.n, _H, _W))
            _all_dap = []
            _all_dap_pos = []

            for opt_id, item in opt_dict.items():
                _all_pos = []
                _all_z_kl = []
                _all_pi_kl = []
                _all_ap = []
                _all_dir = []
                for tau in opt_dict[opt_id]:
                    _pos = tau['agent_pos'][:tau['ep_len']]
                    _dir = tau['agent_dir'][:tau['ep_len']]
                    _all_pos.append(_pos)

                    # _pi_kl = tau['pi_kl_tensor'][:tau['ep_len']]
                    # _all_pi_kl.append(_pi_kl)

                    # if self.args.use_infobot:
                    _z_kl = tau['zz_kl_tensor'][:tau['ep_len']]
                    _all_z_kl.append(_z_kl)
                    _dap = tau['default_action_probs'][:tau['ep_len']]
                    _ap = tau['action_probs'][:tau['ep_len']]
                    if self.action_space_type == 'pov':
                        _ap_cardinal = pov_to_cardinal(_ap, _dir)
                        _dap_cardinal = pov_to_cardinal(_dap, _dir)
                        _all_ap.append(_ap_cardinal)
                        _all_dap.append(_dap_cardinal)
                    else:
                        _all_ap.append(_ap)
                        _all_dap.append(_dap)
                    _all_dap_pos.append(_pos)

                if len(_all_pos) == 0:
                    continue

                _all_pos = np.concatenate(_all_pos, 0)
                _all_z_kl = np.concatenate(_all_z_kl, 0)
                # _all_pi_kl = np.concatenate(_all_pi_kl, 0)
                _all_ap = np.concatenate(_all_ap, 0)

                _pos_map = self.get_hmap(_all_pos)
                _all_z_kl = self.get_hmap(_all_pos, _all_z_kl, freq_normalize=True)
                # _pi_kl_map = self.get_hmap(_all_pos, _all_pi_kl, freq_normalize=True)
                _all_ap = self.get_hmap(_all_pos, _all_ap, freq_normalize=True)

                opt_pos_maps[opt_id] = _pos_map
                opt_z_kl_maps[opt_id] = _all_z_kl
                # opt_pi_kl_maps[opt_id] = _pi_kl_map
                opt_ap[opt_id] = _all_ap

            _all_dap_pos = np.concatenate(_all_dap_pos, 0)
            _all_dap = np.concatenate(_all_dap, 0)
            dap_map = self.get_hmap(_all_dap_pos, _all_dap, freq_normalize=True)

        if self.args.option_space == 'discrete':
            nrow = min(5, omega_dim_current)
            # opt_pos_maps = [utils.normalize_map(np.stack(_opt_map, 0)) \
            #     for _opt_map in opt_pos_maps]
            # opt_z_kl_maps = [utils.normalize_map(np.stack(_opt_map, 0)) \
            #     for _opt_map in opt_z_kl_maps]
            # opt_pi_kl_maps = [utils.normalize_map(np.stack(_opt_map, 0)) \
            #     for _opt_map in opt_pi_kl_maps]
            # opt_ap = [utils.normalize_map(np.stack(_opt_map, 0)) \
            #     for _opt_map in opt_ap]

            opt_pos_titles = ["pos_{}".format(idx) \
                for idx in range(len(opt_pos_maps))]
            opt_z_kl_titles = ["on_z_kl_{}".format(idx) \
                for idx in range(len(opt_pos_maps))]
            opt_pi_kl_titles = ["on_pi_kl_{}".format(idx) \
                for idx in range(len(opt_pos_maps))]

            opt_pos_maps = np.stack(opt_pos_maps, 0)
            opt_z_kl_maps = np.stack(opt_z_kl_maps, 0)
            # opt_pi_kl_maps = np.stack(opt_pi_kl_maps, 0)
            opt_ap = np.stack(opt_ap, 0)
            # opt_z_kl_mean = opt_z_kl_maps.mean(0)
            # opt_pi_kl_mean = opt_pi_kl_maps.mean(0)
            opt_z_kl_mean = opt_z_kl_maps.mean(0)
            opt_z_kl_max = opt_z_kl_maps.max(0)
            opt_pos_mean = opt_pos_maps.mean(0)
            opt_ap_mean = opt_ap.mean(0)

            # Computing pi_kl
            opt_ap_mean_tensor = torch.from_numpy(
                opt_ap_mean.transpose([1, 2, 0])).to(self.device)
            opt_ap_tensor = torch.from_numpy(
                opt_ap.transpose([0, 2, 3, 1])).to(self.device)
            opt_dist_mean = FixedCategorical(probs=opt_ap_mean_tensor)
            pi_kl = make_op_list((_H, _W))
            for opt_id in range(omega_dim_current):
                opt_dist = FixedCategorical(probs=opt_ap_tensor[opt_id])
                pi_kl[opt_id] = ds.kl.kl_divergence(opt_dist, opt_dist_mean)

            opt_pi_kl = torch.stack(pi_kl, 0).cpu().numpy()
            opt_pi_kl_mean = opt_pi_kl.mean(0)

            # NaN values in KL computation
            opt_pi_kl_mean[np.isnan(opt_pi_kl_mean)] = -1e-6
            opt_pi_kl[np.isnan(opt_pi_kl)] = -1e-6

            subtitles_mean = [
                'pi_kl_mean', 'z_kl_mean', 'z_kl_max', 'pos_mean']
            hmaps_mean = [
                opt_pi_kl_mean, opt_z_kl_mean, opt_z_kl_max, opt_pos_mean]

            # self.viz.plotly_grid(
            #     plot_type='heatmap',
            #     hmap_batch=opt_z_kl_maps,
            #     ncols=nrow,
            #     key="on_policy_opt_z_kl",
            #     subplot_titles=opt_z_kl_titles,
            #     normalize_mode=None,
            #     normalize=False,
            # )
            # self.viz.plotly_grid(
            #     plot_type='heatmap',
            #     hmap_batch=opt_pi_kl,
            #     ncols=nrow,
            #     key="on_policy_opt_pi_kl",
            #     subplot_titles=opt_pi_kl_titles,
            #     normalize_mode=None,
            #     normalize=False,
            # )
            info.update({
                'on_policy_opt_pi_kl':{
                    'plot': opt_pi_kl,
                    'subtitles': opt_pi_kl_titles,
                },
                'on_policy_opt_z_kl':{
                    'plot': opt_z_kl_maps,
                    'subtitles': opt_z_kl_titles,
                },
            })

            self.viz.plotly_grid(
                plot_type='heatmap',
                hmap_batch=np.stack(hmaps_mean),
                ncols=len(hmaps_mean),
                key="on_policy_option_marginalized",
                subplot_titles=subtitles_mean,
                normalize_mode=None,
                iter_id=iter_id,
                save_figures=True,
            )

            if self.args.use_infobot:
                self.viz.plotly_quiver_plot(
                    action_probs=dap_map,
                    action_space_type=self.action_space_type,
                    actions=self.vis_env.actions,
                    key="on_policy_pi_with_zprior",
                    bg_img=self.env_rgb_img,
                    iter_id=iter_id,
                    save_figures=True,
                )
            self.viz.plotly_quiver_plot(
                action_probs=opt_ap_mean,
                action_space_type=self.action_space_type,
                actions=self.vis_env.actions,
                key="on_policy_pi_opt_marginalized",
                bg_img=self.env_rgb_img,
                iter_id=iter_id,
                save_figures=True,
            )

            nrow = min(5, omega_dim_current)
            self.viz.plotly_grid(
                plot_type='quiver',
                hmap_batch=opt_ap,
                ncols=nrow,
                key="on_policy_pi",
                action_space_type=self.action_space_type,
                actions=self.vis_env.actions,
                subplot_titles=None,
                normalize_mode=None,
                iter_id=iter_id,
                save_figures=True,
            )

        return info

    def update_on_policy_continuous(self, iter_id, omega_dim_current):
        info = {}
        op_q = self.m_opt_track_q.get_all_items()

        op_q['s_f_pos_velocity'] = op_q['s_f_pos_velocity'].reshape(
            -1, *op_q['s_f_pos_velocity'].shape[2:])

        op_q['omega'] = op_q['omega'].reshape(
            -1, *op_q['omega'].shape[2:])

        op_q['ep_len'] = op_q['ep_len'].reshape(
            -1, *op_q['ep_len'].shape[2:])

        op_q['pos_velocity'] = op_q['pos_velocity'].transpose([0, 2, 1, 3])
        op_q['pos_velocity'] = op_q['pos_velocity'].reshape(
            -1, *op_q['pos_velocity'].shape[2:])

        op_q['masks'] = op_q['masks'].transpose([0, 2, 1, 3])
        op_q['masks'] = op_q['masks'].reshape(
            -1, *op_q['masks'].shape[2:])

        # op_q['z_loc'] = op_q['z_loc'].transpose([0, 2, 1, 3])
        # op_q['z_loc'] = op_q['z_loc'].reshape(
        #     -1, *op_q['z_loc'].shape[2:])
        #
        # op_q['z_std'] = op_q['z_std'].transpose([0, 2, 1, 3])
        # op_q['z_std'] = op_q['z_std'].reshape(
        #     -1, *op_q['z_std'].shape[2:])

        op_q['zz_kl_tensor'] = op_q['zz_kl_tensor'].transpose([0, 2, 1, 3])
        op_q['zz_kl_tensor'] = op_q['zz_kl_tensor'].reshape(
            -1, *op_q['zz_kl_tensor'].shape[2:])

        op_q['pi_kl_tensor'] = op_q['pi_kl_tensor'].transpose([0, 2, 1, 3])
        op_q['pi_kl_tensor'] = op_q['pi_kl_tensor'].reshape(
            -1, *op_q['pi_kl_tensor'].shape[2:])

        op_q['action_probs'] = \
            op_q['action_probs'].transpose([0, 2, 1, 3])
        op_q['action_probs'] = \
            op_q['action_probs'].reshape(
                -1, *op_q['action_probs'].shape[2:])

        op_q['default_action_probs'] = \
            op_q['default_action_probs'].transpose([0, 2, 1, 3])
        op_q['default_action_probs'] = \
            op_q['default_action_probs'].reshape(
                -1, *op_q['default_action_probs'].shape[2:])

        # op_q['action_log_probs'] = \
        #     op_q['action_log_probs'].transpose([0, 2, 1, 3])
        # op_q['action_log_probs'] = \
        #     op_q['action_log_probs'].reshape(
        #         -1, *op_q['action_log_probs'].shape[2:])

        op_q['s_f_pos_velocity'] = op_q['s_f_pos_velocity']
        op_q['ep_len'] = op_q['ep_len'].astype('int')
        op_q['pos_velocity'] = op_q['pos_velocity']
        n_items = op_q['s_f_pos_velocity'].shape[0]

        # GROUP_BY = 's_f'
        assert self.args.option_space == 'discrete'
        GROUP_BY = 'option'
        # else:
        #     GROUP_BY = 's_f'
        # assert GROUP_BY in ['s_f', 'option']

        opt_dict = {key:[] for key in range(omega_dim_current)}
        for idx in range(n_items):
            _opt_id = op_q['omega'][idx].argmax()
            qu_item = {key: item[idx] for key, item in op_q.items()}
            opt_dict[_opt_id].append(qu_item)

        # for opt_id in opt_dict.keys():
        #     for key in opt_dict[opt_id][0].keys():
        #         item = [opt_dict[opt_id][j][key] \
        #             for j in range(len(opt_dict[opt_id]))]
        #         opt_dict[key] = np.stack(item, 0)

        # def pov_to_cardinal(action_probs, dir):
        #     ap_cardinal = np.zeros((action_probs.shape[0], 4 + 6))
        #     fwd_prob = action_probs[:, self.vis_env.actions.forward]
        #     left_prob = action_probs[:, self.vis_env.actions.left]
        #     right_prob = action_probs[:, self.vis_env.actions.right]
        #     # _other_prob = action_probs[:,
        #     #     np.arange(action_probs.shape[1]) != self.vis_env.actions.forward]
        #     ap_cardinal[np.arange(action_probs.shape[0]),
        #         dir[:, 0]] = fwd_prob
        #     ap_cardinal[:, 4] = left_prob
        #     ap_cardinal[:, 5] = right_prob
        #     ap_cardinal[:, 6:] = action_probs[:, 3:]
        #     # _dap_map = self.get_hmap(_pos, ap_cardinal, freq_normalize=True)
        #     return ap_cardinal

        # _H = self.vis_env.height
        # _W = self.vis_env.width
        make_op_list = lambda: [None \
            for _ in range(omega_dim_current)]
        opt_posvel_maps = make_op_list()
        opt_final_posvel_maps = make_op_list()
        opt_z_kl_maps = make_op_list()
        th_opt_posvel_maps = make_op_list()
        th_opt_z_kl_maps = make_op_list()
        # # opt_pi_kl_maps = make_op_list()
        # if self.action_space_type == 'pov':
        #     opt_ap = make_op_list((4 + 6, _H, _W))
        # else:
        #     opt_ap = make_op_list(
        #         (self.vis_env.action_space.n, _H, _W))
        # # _all_dap = []
        # # _all_dap_pos = []

        def get_scatter(xy, z_value=None):
            _dict = {'x': xy[:, 0], 'y': xy[:, 1]}
            if z_value is not None:
                _dict['z'] = np.squeeze(z_value)
            return _dict

        # Top K% threshold
        _THRESHOLD = 0.025
        _TH_MIN_COUNT = 3

        for opt_id, item in opt_dict.items():
            _all_posvel = []
            _all_final_posvel = []
            _all_z_kl = []
            _th_all_posvel = []
            _th_all_z_kl = []
            _all_pi_kl = []
            # _all_ap = []
            for tau in opt_dict[opt_id]:
                min_count = min(tau['ep_len'], _TH_MIN_COUNT)
                n_top = max(int(tau['ep_len'] * _THRESHOLD), min_count)
                _posvel = tau['pos_velocity'][:tau['ep_len']]
                _z_kl = tau['zz_kl_tensor'][:tau['ep_len']]

                _inds = np.argpartition(np.squeeze(_z_kl), -n_top)[-n_top:]
                _th_z_kl = _z_kl[_inds]
                _th_posvel = _posvel[_inds]

                _all_posvel.append(_posvel)
                _all_z_kl.append(_z_kl)
                _final_posvel = _posvel[-1:]
                _all_final_posvel.append(_final_posvel)

                _th_all_posvel.append(_th_posvel)
                _th_all_z_kl.append(_th_z_kl)

                # _dap = tau['default_action_probs'][:tau['ep_len']]
                # _ap = tau['action_probs'][:tau['ep_len']]

            if len(_all_posvel) == 0:
                continue

            _all_posvel = np.concatenate(_all_posvel, 0)
            _all_z_kl = np.concatenate(_all_z_kl, 0)
            _all_final_posvel = np.concatenate(_all_final_posvel, 0)
            _th_all_posvel = np.concatenate(_th_all_posvel, 0)
            _th_all_z_kl = np.concatenate(_th_all_z_kl, 0)
            # _all_pi_kl = np.concatenate(_all_pi_kl, 0)

            posvel_sdict = get_scatter(_all_posvel)
            th_posvel_sdict = get_scatter(_th_all_posvel)
            final_posvel_sdict = get_scatter(_all_final_posvel)
            z_kl_sdict = get_scatter(_all_posvel, _all_z_kl)
            th_z_kl_sdict = get_scatter(_th_all_posvel, _th_all_z_kl)

            # _pi_kl_map = self.get_hmap(_all_pos, _all_pi_kl, freq_normalize=True)

            opt_posvel_maps[opt_id] = posvel_sdict
            opt_z_kl_maps[opt_id] = z_kl_sdict
            opt_final_posvel_maps[opt_id] = final_posvel_sdict
            # opt_pi_kl_maps[opt_id] = _pi_kl_map
            # opt_ap[opt_id] = _all_ap

            th_opt_posvel_maps[opt_id] = th_posvel_sdict
            th_opt_z_kl_maps[opt_id] = th_z_kl_sdict

        # _all_dap_pos = np.concatenate(_all_dap_pos, 0)
        # _all_dap = np.concatenate(_all_dap, 0)
        # dap_map = self.get_hmap(_all_dap_pos, _all_dap, freq_normalize=True)

        opt_posvel_maps = {f'opt_{idx}':val \
            for idx, val in enumerate(opt_posvel_maps)}

        opt_final_posvel_maps = {f'opt_{idx}':val \
            for idx, val in enumerate(opt_final_posvel_maps)}

        opt_z_kl_maps = {f'opt_{idx}':val \
            for idx, val in enumerate(opt_z_kl_maps)}

        th_opt_posvel_maps = {f'opt_{idx}':val \
            for idx, val in enumerate(th_opt_posvel_maps)}

        th_opt_z_kl_maps = {f'opt_{idx}':val \
            for idx, val in enumerate(th_opt_z_kl_maps)}

        self.viz.plotly_scatter(
            opt_posvel_maps,
            key='opt_posvel_maps',
        )

        self.viz.plotly_scatter(
            th_opt_posvel_maps,
            key='th_opt_posvel_maps',
        )

        self.viz.plotly_scatter(
            opt_final_posvel_maps,
            key='opt_final_posvel_maps',
        )

        self.viz.plotly_scatter(
            opt_z_kl_maps,
            key='opt_z_kl_maps',
        )

        self.viz.plotly_scatter(
            th_opt_z_kl_maps,
            key='th_opt_z_kl_maps',
        )

        self.viz.plotly_scatter(
            opt_z_kl_maps,
            key='opt_z_kl_maps_normalized',
            normalize=True,
        )

        self.viz.plotly_scatter(
            th_opt_z_kl_maps,
            key='th_opt_z_kl_maps_normalized',
            normalize=True,
        )

        self.viz.plotly_scatter(
            opt_z_kl_maps,
            key='opt_z_kl_maps_opacity',
            z_effect='opacity',
        )

        self.viz.plotly_scatter(
            th_opt_z_kl_maps,
            key='th_opt_z_kl_maps_opacity',
            z_effect='opacity',
        )

        self.viz.plotly_mc_viz(
            th_opt_z_kl_maps,
            key=f'opt_z_kl_TH-{_THRESHOLD}',
            normalize=True,
        )


        # self.viz.plotly_scatter(
        #     opt_z_kl_maps,
        #     key='opt_z_kl_maps_opacity_log',
        #     z_effect='opacity',
        #     opacity_scaling='log',
        # )

        # opt_z_kl_th = {}
        # for opt_id, opt_dict in opt_z_kl_maps.items():
        #     opt_z_kl_th[opt_id] = {}
        #     n_top = int(len(opt_dict['z']) * _THRESHOLD)
        #     inds = np.argpartition(opt_dict['z'], -n_top)[-n_top:]
        #     # print(opt_z_kl_maps[opt_id]['z'].min())
        #     for key in opt_dict:
        #         opt_z_kl_th[opt_id][key] = \
        #             opt_z_kl_maps[opt_id][key][inds]
        #     # print(opt_z_kl_maps[opt_id]['z'].min())
        #     # print("Wait!")
        #
        # self.viz.plotly_scatter(
        #     opt_posvel_maps,
        #     key=f'opt_posvel_TH{_THRESHOLD}_maps',
        # )
        #
        # self.viz.plotly_scatter(
        #     opt_z_kl_th,
        #     key=f'opt_z_kl_TH{_THRESHOLD}_normalized',
        #     normalize=True,
        # )
        #
        # self.viz.plotly_scatter(
        #     opt_z_kl_th,
        #     key=f'opt_z_TH{_THRESHOLD}_maps_opacity',
        #     z_effect='opacity',
        # )
        #
        # self.viz.plotly_scatter(
        #     opt_z_kl_th,
        #     key=f'opt_z_TH{_THRESHOLD}_maps_opacity_log',
        #     z_effect='opacity',
        #     opacity_scaling='log',
        # )
        #
        # self.viz.plotly_mc_viz(
        #     opt_z_kl_maps,
        #     key=f'opt_z_kl_TH-{_THRESHOLD}',
        #     normalize=True,
        # )
        return info

    def update_heatmaps(self, rollouts, omega_option, omega_dim_current,
        action_dist_probs):

        masks = rollouts.masks
        if self.args.is_pomdp_env and self.args.model == 'cond':
        # if self.args.is_pomdp_env:
            plus_masks = torch.eq(torch.cat([masks[0:1], masks[:-1]], 0), 1)
            masks = plus_masks

        final_step_indices = masks.sum(0).long() - 1
        final_step_indices = final_step_indices.repeat(
            1, self.agent_pos.shape[2]).unsqueeze(0)

        pos_tensor = torch.from_numpy(self.agent_pos).to(self.device)
        final_step_pos = pos_tensor.gather(
            dim=0, index=final_step_indices)[0]

        initial_pos = pos_tensor[0]

        all_x = pos_tensor[:, :, :1].masked_select(
            masks.ne(0))
        all_y = pos_tensor[:, :, 1:].masked_select(
            masks.ne(0))
        all_step_pos = torch.stack([all_x, all_y], 1)

        pos_init = self.get_hmap(initial_pos)
        pos_final = self.get_hmap(final_step_pos)
        pos_all = self.get_hmap(all_step_pos)
        self.m_istate_hmap.add(pos_init)
        self.m_fstate_hmap.add(pos_final)
        self.m_all_steps_hmap.add(pos_all)

        if self.args.model != 'cond' \
        and self.args.option_space == 'discrete'\
        and self.args.hier_mode != 'infobot-supervised' \
        and self.args.env_name != 'crossing':
            action_dist_probs = torch.from_numpy(action_dist_probs).to(self.device)
            action_dims = action_dist_probs.shape[-1]
            # all_ap = action_dist_probs.new_zeros((all_x.shape[0], action_dims))
            # for action_dim in range(action_dims):
            #     all_ap[:, action_dim] = \
            #         action_dist_probs[:, :, action_dim:action_dim + 1].masked_select(
            #             masks[:-1].ne(0))
            # pos_ap = self.get_hmap(all_step_pos, all_ap)

            # m_option_choice.add(option_one_hot.mean(0))
            # if (goal_index==0).sum() > 0:
            #     m_option_g1.add(option_one_hot[goal_index==0].mean(0))
            # if (goal_index==1).sum() > 0:
            #     m_option_g2.add(option_one_hot[goal_index==1].mean(0))

            opt_ap_dict = {}
            final_step_pos = final_step_pos.cpu().numpy()
            for opt_idx in range(self.args.omega_option_dims):
                option_np = omega_option.cpu().numpy().argmax(1)
                _inds = np.where(option_np == opt_idx)[0]
                if len(_inds) == 0:
                    continue
                _final_step = final_step_pos[_inds]
                _pos_final = self.get_hmap(_final_step)

                _all_x = pos_tensor[:, _inds, :1].masked_select(
                    masks[:, _inds].ne(0))
                _all_y = pos_tensor[:, _inds, 1:].masked_select(
                    masks[:, _inds].ne(0))
                _all_step_pos = torch.stack([_all_x, _all_y], 1)

                # Direction
                agent_dir = torch.from_numpy(self.agent_dir).to(self.device)
                _all_dir = agent_dir[:, _inds].masked_select(
                    masks[:, _inds].ne(0))

                # Action probs
                _all_ap = action_dist_probs.new_zeros((_all_x.shape[0], action_dims))
                for a_dim in range(action_dims):
                    _all_ap[:, a_dim] = \
                        action_dist_probs[:, _inds, a_dim:a_dim + 1].masked_select(
                            masks[:-1, _inds].ne(0))

                if self.action_space_type == 'pov':
                    _dap_cardinal = np.zeros((_all_ap.shape[0], 4 + 4))
                    _fwd_prob = _all_ap[:, self.vis_env.actions.forward]
                    _dap_cardinal[np.arange(_all_ap.shape[0]), _all_dir] = _fwd_prob
                    _dap_cardinal[:, 4:] = _all_ap[:, 3:]
                    # _dap_cardinal = np.zeros((_dap.shape[0], 4 + 4))
                    # _fwd_prob = _dap[:, self.vis_env.actions.forward]
                    # # _other_prob = _dap[:,
                    # #     np.arange(_dap.shape[1]) != self.vis_env.actions.forward]
                    # _dap_cardinal[np.arange(_dap.shape[0]),
                    #     _dir[:, 0]] = _fwd_prob
                    # _dap_cardinal[:, 4:] = _dap[:, 3:]
                    # _dap_map = self.get_hmap(_pos, _dap_cardinal, freq_normalize=True)
                    _pos_ap = self.get_hmap(_all_step_pos, _dap_cardinal)
                else:
                    _pos_ap = self.get_hmap(_all_step_pos, _all_ap)

                _pos_all = self.get_hmap(_all_step_pos)
                opt_ap_dict[opt_idx] = _pos_ap

                self.m_options['final_state'][opt_idx].add(_pos_final)
                self.m_options['all_states'][opt_idx].add(_pos_all)
                # self.m_options['pi_def'][opt_idx].add(_pos_ap)

            pos_pi_def = np.stack(opt_ap_dict.values(), 0)
            # _pos_ap_all_options = pos_pi_def.sum(0) / omega_dim_current
            pos_pi_def = utils.normalize_map(pos_pi_def)
            self.m_pi_def.add(pos_pi_def)

            EPSILON = 1e-8
            pos_pi_def = torch.from_numpy(
                pos_pi_def.transpose([1, 2, 0])).to(self.device)
            pos_pi_def_dist = FixedCategorical(probs=pos_pi_def + EPSILON)
            kl_dict = {}
            for opt_idx, opt_pos_ap in opt_ap_dict.items():
                pi_opt = torch.from_numpy(
                    opt_pos_ap.transpose([1, 2, 0])).to(self.device)
                pi_opt_dist = FixedCategorical(probs=pi_opt + EPSILON)
                kl_pi_def = ds.kl.kl_divergence(pi_opt_dist, pos_pi_def_dist)
                kl_pi_def = kl_pi_def.cpu().numpy()
                # Note: NaN value issue below is resolved
                # # kl_pi_def = np.nan_to_num(kl_pi_def)
                # # NOTE: Nan values occur when pi not defiend (=0) on
                # # state s_t for the given option
                # kl_pi_def[np.isnan(kl_pi_def)] = 0.0
                assert np.isnan(kl_pi_def).sum().item() == 0
                kl_pi_def = kl_pi_def * (kl_pi_def > 0)
                kl_dict[opt_idx] = kl_pi_def
                self.m_options['pi_def_kl'][opt_idx].add(kl_pi_def)

            avg_kl_pidef = np.stack(kl_dict.values(), 0)
            avg_kl_pidef = avg_kl_pidef * (avg_kl_pidef > 0)
            avg_kl_pidef = utils.normalize_map(avg_kl_pidef)
            self.m_pi_def_kl.add(avg_kl_pidef)

    def update_visdom_plots(
        self,
        start_iter,
        iter_id,
        total_time_steps,
        return_mean,
        dist_entropy,
        cpu_episode_rewards,
        omega_dim_current,
        omega_dim_ll_threshold,
        ic_info,
    ):

        # Unpack ic_info
        anneal_coeff = ic_info['anneal_coeff']
        infobot_coeff = ic_info['infobot_coeff']
        q_start_flag = ic_info['q_start_flag']
        KLD = ic_info['kld_qp']
        p_ll = ic_info['p_ll']
        q_ll = ic_info['q_ll']
        p_entropy = ic_info['p_entropy']
        q_entropy = ic_info['q_entropy']
        pq_loss = ic_info['pq_loss']
        log_det_j_mean = ic_info['log_det_j']
        batch_weights_entropy = np.squeeze(ic_info['batch_weights_entropy'])
        # path_derivative_loss = ic_info['path_derivative_loss']

        # print(options_decoder.fc[0].weight.sum().item())
        if self.args.use_infobot:
            z_entropy = ic_info['z_entropy']
            zz_kl_loss = ic_info['zz_kl_loss']
            zz_kld = ic_info['zz_kld']
            # zz_kl_tensor = ic_info['zz_kl_tensor']
            # zz_lld_tensor = ic_info['zz_lld_tensor']

        # Measuring iterations vs epochs
        self.viz.line(iter_id, iter_id / self.num_batches_per_epoch,
            "epochs vs iterations", "epochs")
        self.viz.line(iter_id, total_time_steps // 1000,
            "epochs vs iterations", "total_time_steps x 1e-4")

        total_episodes = (iter_id - start_iter + 1) * self.num_processes_eff
        self.viz.line(iter_id, int(total_episodes / (self.end_t - self.start_t)),
           "Training speed in episodes/sec", "episodes/sec")

        # Return
        eff_ret_avg, eff_ret_std = self.m_eff_return.value()
        self.viz.line(iter_id,  eff_ret_avg, "effective_return", "smooth_return")
        # self.viz.line(iter_id,  eff_ret_avg + eff_ret_std,
        #     "effective_return", "smooth_up", dash="dash")
        # self.viz.line(iter_id,  eff_ret_avg - eff_ret_std,
        #     "effective_return", "smooth_down", dash="dash")

        # return_se = scipy.stats.sem(cpu_episode_rewards, axis=None)
        self.viz.line(iter_id, return_mean , "return", "return_mean")
        # return_mov_avg, reward_std = self.m_reward.value()
        return_moments = self.m_reward.value()
        return_mov_avg, reward_std = return_moments['mean']
        # self.viz.line(iter_id, moving_avg_reward , "return", "train_mean_smooth")
        self.viz.line(iter_id,  return_mov_avg, "return", "return_mean_smooth")
        # self.viz.line(iter_id,  return_mov_avg + reward_std,
        #     "return", "smooth_up", dash="dash")
        # self.viz.line(iter_id,  return_mov_avg - reward_std,
        #     "return", "smooth_down", dash="dash")
        # self.viz.line(iter_id, np.median(cpu_episode_rewards), "return", "train_median")
        self.viz.line(iter_id, cpu_episode_rewards.min(), "return", "train_min")
        self.viz.line(iter_id, cpu_episode_rewards.max(), "return", "train_max")

        # self.viz.line(iter_id, return_moments['median'][0], "return", "train_median")
        # self.viz.line(iter_id, return_moments['min'][0], "return", "train_min")
        # self.viz.line(iter_id, return_moments['max'][0], "return", "train_max")


        self.viz.line(iter_id, self.m_value_pred.value()[0],
            "effective_return", "value_pred_t0")
        self.viz.line(iter_id, self.m_empowerment.value()[0],
            "effective_return", "empowerment")

        self.viz.line(iter_id, self.m_bonus_reward.value()[0],
            "effective_return", "visitation_bonus")

        visit_bonus_mean_std = self.m_bonus_reward.value()
        self.viz.line(iter_id, visit_bonus_mean_std[0],
            "visit_bonus", "mean")
        # self.viz.line(iter_id, visit_bonus_mean_std[0] + visit_bonus_mean_std[1],
        #     "visit_bonus", "mean_up")
        # self.viz.line(iter_id, visit_bonus_mean_std[0] - visit_bonus_mean_std[1],
        #     "visit_bonus", "mean_down")
        self.viz.line(iter_id, self.m_bonus_std.value()[0],
            "visit_bonus", "std")

        if self.args.ic_mode == 'diyan':
            self.viz.line(iter_id, self.m_empowerment_sum_t.value()[0],
                "effective_return", "empowerment_sum_t")


        self.viz.line(iter_id, dist_entropy, "dist_entropy", "entropy")
        dist_entropy_avg, _ = self.m_dist_entropy.value()
        self.viz.line(iter_id, dist_entropy_avg, "dist_entropy", "smooth")

        # self.viz.line(iter_id, value_loss, "value_loss", "loss")
        value_mov_avg, value_std = self.m_value_loss.value()
        self.viz.line(iter_id, value_mov_avg,
            "value_loss", "smooth_value_loss")
        self.viz.line(iter_id, value_mov_avg + value_std,
            "value_loss", "smooth_up", dash="dash")
        self.viz.line(iter_id, value_mov_avg - value_std,
            "value_loss", "smooth_down", dash="dash")

        # self.viz.line(iter_id, action_loss, "action_loss", "loss")
        action_loss_avg, action_loss_std = self.m_action_loss.value()
        self.viz.line(iter_id, action_loss_avg,
            "action_loss", "smooth_action_loss")
        self.viz.line(iter_id, action_loss_avg + action_loss_std,
            "action_loss", "smooth-up", dash="dash")
        self.viz.line(iter_id, action_loss_avg - action_loss_std,
            "action_loss", "smooth-down", dash="dash")

        # pathd_loss_avg, pathd_loss_std = m_pathd_loss.value()
        # self.viz.line(iter_id, pathd_loss_avg, "path_derivative_loss", "smooth")
        # self.viz.line(iter_id, pathd_loss_avg + pathd_loss_std,
        #     "path_derivative_loss", "smooth-up", dash="dash")
        # self.viz.line(iter_id, pathd_loss_avg - pathd_loss_std,
        #     "path_derivative_loss", "smooth-down", dash="dash")

        # # self.viz.line(iter_id, action_log_probs_mean,
        # #     "action_log_probs", "alp")
        # alp_avg, alp_std = self.m_alp_mean.value()
        # self.viz.line(iter_id, -alp_avg * self.args.max_ent_action_logprob_coeff,
        #     "action_log_probs", "smooth_alp_loss")
        # self.viz.line(iter_id, alp_avg, "action_log_probs", "smooth_alp")
        # self.viz.line(iter_id, self.args.max_ent_action_logprob_coeff,
        #     "action_log_probs", "coeff")

        if self.args.allow_early_stop:
           ep_len = self.step_counter.astype('int') #.sum(0)
           ep_len += 1 # First action counting
           ep_key = "Episode length"
           m_avg_eplen, _ = self.m_episode_len.value()
           self.viz.line(iter_id, ep_len.mean(), ep_key, "train_mean")
           self.viz.line(iter_id, m_avg_eplen, ep_key, "smooth")
           self.viz.line(iter_id, np.median(ep_len), ep_key, "train_median")
           self.viz.line(iter_id, ep_len.min(), ep_key, "train_min")
           self.viz.line(iter_id, ep_len.max(), ep_key, "train_max")

        # KL Plots
        if self.args.model == 'hier':
            if self.args.hier_mode != 'infobot-supervised':
                if self.args.option_space == 'discrete':
                    self.viz.line(iter_id, omega_dim_current,
                        "Omega dim curriculum", "omega_dims")
                    self.viz.line(iter_id, self.args.omega_option_dims,
                        "Omega dim curriculum", "max_dims")

                self.viz.line(iter_id, KLD, "KL Plots", "KLD_q_p")
                self.viz.line(iter_id, self.args.hr_model_kl_coeff * anneal_coeff,
                    "KL Plots", "kl_coeff")
                # self.viz.line(iter_id, kl_loss, "KL Plots", "kl_loss")
                self.viz.line(iter_id, pq_loss, "KL Plots", "pq_loss")

                # if self.args.closed_loop:
                    # self.viz.line(iter_id, JSD, "KL Plots", "JSD")
                self.viz.line(iter_id, int(q_start_flag), "KL Plots", "q_start")
                self.viz.line(iter_id, p_ll, "PQ LL", "p_ll")
                self.viz.line(iter_id, self.m_traj_enc_ll.value()[0], "PQ LL", "p_ll_smooth")
                self.viz.line(iter_id, omega_dim_ll_threshold, "PQ LL", "p_ll_threshold")
                self.viz.line(iter_id, q_ll, "PQ LL", "q_ll")
                # self.viz.line(iter_id, p_entropy, "PQ entropy", "p_entropy")
                # self.viz.line(iter_id, q_entropy, "PQ entropy", "q_entropy")
                # self.viz.line(iter_id, batch_weights_entropy,
                #     "PQ entropy", "batch_wt_entropy")
                self.viz.line(iter_id, log_det_j_mean, "KL Plots", "ldj_mean")

                # Option loss
                option_loss_avg, option_loss_std = self.m_option_loss.value()
                self.viz.line(iter_id, option_loss_avg, "action_loss", "option_loss")
                # self.viz.line(iter_id, option_loss_avg + option_loss_std,
                #     "option_loss", "smooth-up", dash="dash")
                # self.viz.line(iter_id, option_loss_avg - option_loss_std,
                #     "option_loss", "smooth-down", dash="dash")

            if self.args.use_infobot:
                self.viz.line(iter_id, self.args.infobot_beta * infobot_coeff,
                    "zz_KL", "zz_kl_coeff")
                self.viz.line(iter_id, zz_kld, "zz_KL", "zz_kld")
                self.viz.line(iter_id, zz_kl_loss, "zz_KL", "zz_kl_loss")
                self.viz.line(iter_id, z_entropy, "zz_KL", "z_entropy")

            if self.args.hier_mode == 'transfer':
                self.viz.line(iter_id, self.m_opt_value_loss.value()[0],
                    "value_loss", "option_loss")
                self.viz.line(iter_id, self.m_opt_returns.value()[0],
                    "return", "opt_returns")

    def update_visdom_success_plot(self, iter_id, success, max_room_id, vis_info):
        if not hasattr(self, 'best_success'):
            self.best_success = 0.0
        self.best_success = max(self.best_success, success.mean())

        if not hasattr(self, 'best_max_room_id'):
            self.best_max_room_id = 0.0
        self.best_max_room_id = max(self.best_max_room_id, max_room_id.mean())

        self.viz.line(iter_id, success.mean(), "success", "mean")
        self.viz.line(iter_id, self.best_success, "success", "best")
        # self.viz.line(iter_id, success.max(), "success", "max")
        # self.viz.line(iter_id, success.min(), "success", "min")
        # self.viz.line(iter_id, np.median(success), "success", "median")

        self.viz.line(iter_id, max_room_id.mean(), "max_room_id", "mean")
        # self.viz.line(iter_id, max_room_id.max(), "max_room_id", "max")
        self.viz.line(iter_id, self.best_max_room_id, "max_room_id", "best")

        bonus_kl_grid = vis_info['kl_grid']
        bonus_kl_grid_avg = vis_info['kl_grid_avg']
        bonus_isq_grid = vis_info['isq_count_grid']
        bonus_isq_grid_avg = vis_info['isq_count_grid_avg']
        bonus_grid = vis_info['bonus_grid']
        bonus_grid_avg = vis_info['bonus_grid_avg']
        t_count = vis_info['t_count']
        # mask = (self.vis_env._occupancy_grid == 0).T
        # mask = (mask * 0) + 1
        mask = 1
        self.viz.heatmap(
            t_count * mask,
            key='visit_count',
            root_power=1,
        )
        # self.viz.heatmap(
        #     bonus_isq_grid * mask,
        #     key='isq_count_grid',
        #     root_power=1,
        # )
        self.viz.heatmap(
            bonus_isq_grid_avg * mask,
            key='isq_count_grid_avg',
            root_power=1,
        )
        self.viz.heatmap(
            bonus_grid * mask,
            key='bonus_grid',
            root_power=1,
        )
        self.viz.heatmap(
            bonus_grid_avg * mask,
            key='bonus_grid_avg',
            root_power=1,
        )
        if self.args.bonus_type == 'kl':
            # self.viz.heatmap(
            #     bonus_kl_grid * mask,
            #     key='bonus_kl_grid',
            #     root_power=1,
            # )
            self.viz.heatmap(
                bonus_kl_grid_avg,
                key='bonus_kl_grid_avg',
                root_power=1,
            )

        self.viz.image(vis_info['rgb_env_image'], 'eval_env')

    def update_visdom_heatmaps(self, iter_id, omega_option, rollouts,
        omega_dim_current, ic_info):

        bonus_tensor=None
        if 'bonus_tensor' in ic_info:
            bonus_tensor = ic_info['bonus_tensor']

        self.viz.action_table(
            key="actions_taken",
            actions=rollouts.actions.cpu().numpy(),
            action_class=self.vis_env.actions,
            agent_pos=self.agent_pos,
            reward=rollouts.rewards.cpu().numpy(),
            masks=rollouts.masks.cpu().numpy(),
            bonus_tensor=bonus_tensor,
        )

        if self.args.use_infobot:
            zz_kl_tensor = ic_info['zz_kl_tensor']
            zz_lld_tensor = ic_info['zz_lld_tensor']

        state_maps = [
            self.m_istate_hmap.value(),
            self.m_fstate_hmap.value(),
            self.m_all_steps_hmap.value(),
            # self.m_pi_def_kl.value(),
        ]
        s_titles = [
            'initial_state',
            'final_state',
            'all_steps',
            # 'pi_def_kl',
        ]

        # pi_def = self.m_pi_def.value()
        # self.viz.plotly_quiver_plot(
        #     action_probs=pi_def,
        #     action_space_type=self.action_space_type,
        #     actions=self.vis_env.actions,
        #     key="pi_def",
        #     bg_img=self.env_rgb_img,
        # )
        # self.viz.plotly_quiver_plot

        # self.viz.heatmap(m_istate_hmap.value(), "initial_state", 2)
        # self.viz.heatmap(m_fstate_hmap.value(), "final_state", 2)
        # self.viz.heatmap(m_all_steps_hmap.value(), "all_steps", 2)
        self.viz.plotly_grid(
            'heatmap',
            np.stack(state_maps, 0),
            ncols=len(state_maps),
            key="state_maps",
            subplot_titles=s_titles,
            normalize_mode=None,
            iter_id=iter_id,
            save_figures=True,
        )

        zz_kld = ic_info['zz_kld']
        final_state_counts = self.m_fstate_hmap.value().reshape(-1)
        final_state_prob = final_state_counts / final_state_counts.sum(0)
        final_state_entropy = scipy.stats.entropy(final_state_prob)
        final_state_max_entropy = np.log(final_state_prob.size)

        self.viz.line(iter_id, final_state_entropy,
            "s_f entropy info", "H-s_f")
        self.viz.line(iter_id, final_state_max_entropy,
            "s_f entropy info", "H-s_f-max")
        self.viz.line(iter_id, self.m_empowerment.value()[0],
            "s_f entropy info", "empowerment")
        self.viz.line(iter_id, zz_kld * self.args.num_steps,
            "s_f entropy info", "z_kld_sum_t")

        self.viz.line(iter_id,
            final_state_entropy - (zz_kld * self.args.num_steps),
            "s_f entropy info", "H-s_f_Omega_lb")

        self.viz.line(iter_id,
            final_state_entropy - self.m_empowerment.value()[0],
            "s_f entropy info", "H-s_f_Omega_ub")

        # if self.args.hier_mode == 'bonus':
        #     bonus_kl_grid = ic_info['bonus_kl_grid']
        #     bonus_isq_grid = ic_info['bonus_isq_grid']
        #     bonus_grid = ic_info['bonus_grid']
        #     # mask = (self.vis_env._occupancy_grid == 0).T
        #     # mask = (mask * 0) + 1
        #     mask = 1
        #     self.viz.heatmap(
        #         bonus_isq_grid * mask,
        #         key='inv_sqrt_c_bonus',
        #         root_power=1,
        #     )
        #     self.viz.heatmap(
        #         bonus_grid * mask,
        #         key='bonus_value',
        #         root_power=1,
        #     )
        #     if self.args.bonus_type == 'kl':
        #         self.viz.heatmap(
        #             bonus_kl_grid * mask,
        #             key='z_kl_value',
        #             root_power=1,
        #         )
        #         self.viz.heatmap(
        #             bonus_isq_grid * bonus_kl_grid * mask,
        #             key='kl_count_bonus',
        #             root_power=1,
        #         )

        if self.args.model == 'cond':
            return

        if self.args.option_space == 'discrete' \
        and self.args.hier_mode != 'infobot-supervised':
            final_state_opts = np.stack(
                [item.value() for item in self.m_options['final_state']],
                    0)[:omega_dim_current]
            f_subtitles = ['f_{}'.format(i) \
                for i in range(final_state_opts.shape[0])]

            all_state_opts = np.stack(
                [item.value() for item in self.m_options['all_states']],
                    0)[:omega_dim_current]
            s_subtitles = ['s_{}'.format(i) \
                for i in range(all_state_opts.shape[0])]

            # all_pi_kl = np.stack(
            #     [item.value() for item in self.m_options['pi_def_kl']],
            #         0)[:omega_dim_current]

            # all_pi_kl_whitened = (all_pi_kl - all_pi_kl.mean(0)) \
            #     / all_pi_kl.std(0, keepdims=True, ddof=1)

            # nrow = min(5, omega_dim_current)
            nrow = final_state_opts.shape[0]
            all_plots = np.concatenate([final_state_opts, all_state_opts], 0)
            all_subtitles = f_subtitles + s_subtitles

            # self.viz.plotly_grid('heatmap', final_state_opts,
            #     ncols=nrow, key="opt_f")
            # self.viz.plotly_grid('heatmap', all_state_opts,
            #     ncols=nrow, key="opt_all")
            self.viz.plotly_grid(
                'heatmap',
                hmap_batch=all_plots,
                subplot_titles=all_subtitles,
                ncols=nrow,
                key="opt_state_heatmaps",
                iter_id=iter_id,
                save_figures=True,
            )

            # self.viz.plotly_grid('heatmap', all_pi_kl,
            #     ncols=nrow, key="opt_pi_kl", normalize=False)
            # self.viz.plotly_grid('heatmap', all_pi_kl_whitened,
            #     ncols=nrow, key="opt_pi_kl_whitened", normalize=False)

        if self.args.use_infobot:
            cpu_masks = rollouts.masks.cpu().numpy()
            _pad = np.ones(cpu_masks.sum(0, keepdims=True).shape)
            cpu_masks = np.concatenate([_pad ,cpu_masks])
            cpu_masks = cpu_masks[:-1]
            if self.args.option_space == 'discrete' \
            and self.args.hier_mode != 'infobot-supervised':
                option_np = omega_option.cpu().numpy().argmax(1)
                _options = option_np
            else:
                _options = None
            self.viz.kl_table_and_hmap(
                key="z_kl_table",
                agent_pos=self.agent_pos,
                kl_values=zz_kl_tensor,
                lld_values=zz_lld_tensor,
                masks=cpu_masks,
                options=_options,
            )

    def update_off_policy_plots(self, iter_id, eval_info):
        info = {}
        env_rgb_img = eval_info['env_rgb_img']
        pi_opt_grid = eval_info['pi_opt_grid']
        pi_def_grid = eval_info['pi_def_grid']
        kl_zz_grid = eval_info['kl_zz_grid'] #[np.newaxis, :]
        kl_zz_opt_grid = eval_info['kl_zz_opt_grid']
        kl_pi_grid = eval_info['kl_pi_def_grid']
        kl_pi_opt_grid = eval_info['kl_pi_opt_grid']
        omega_dim_current = kl_pi_opt_grid.shape[0]
        pi_opt_subtitles = ["pi_kl_{}".format(i) \
            for i in range(omega_dim_current)]

        # self.viz.plotly_heatmap(kl_zz_grid, key="off_policy_z_kl",
        #     normalize=False, colorscale='Reds')
        # self.viz.plotly_heatmap(kl_pi_grid, key="off_policy_pi_kl",
        #     normalize=False, colorscale='Reds')
        all_plots = [kl_zz_grid, kl_pi_grid]
        all_subtitles = ['off_policy_z_kl', 'off_policy_pi_kl']
        self.viz.plotly_grid(
            'heatmap',
            np.stack(all_plots),
            ncols=len(all_plots),
            subplot_titles=all_subtitles,
            key="off_policy_z_and_pi_kl",
            normalize=False,
            colorscale='Reds',
        )

        # nrow = min(5, kl_pi_opt_grid.shape[0])
        nrow = kl_pi_opt_grid.shape[0]
        pi_subtitles = ['off_pi_kl_{}'.format(i) for i in range(omega_dim_current)]
        z_subtitles = ['off_z_kl_{}'.format(i) for i in range(omega_dim_current)]
        # all_plots = np.concatenate([kl_pi_opt_grid, kl_zz_opt_grid])
        # all_subtitles = pi_subtitles + z_subtitles
        info.update({
            'off_policy_opt_pi_kl':{
                'plot': kl_pi_opt_grid,
                'subtitles': pi_subtitles,
            },
            'off_policy_opt_z_kl':{
                'plot': kl_zz_opt_grid,
                'subtitles': z_subtitles
            }
        })

        # self.viz.plotly_grid('heatmap', kl_pi_opt_grid,
        #     subplot_titles=pi_opt_subtitles,
        #     ncols=nrow, key="off_policy_opt_pi_kl", normalize=False)
        # self.viz.plotly_grid('heatmap',
        #     all_plots,
        #     subplot_titles=all_subtitles,
        #     ncols=nrow,
        #     key="off_policy_kl",
        #     normalize=False)

        self.viz.plotly_quiver_plot(
            action_probs=pi_def_grid,
            action_space_type=self.action_space_type,
            actions=self.vis_env.actions,
            key="off_policy_pi_opt_marginalized",
            bg_img=env_rgb_img,
            iter_id=iter_id,
            save_figures=True,
        )

        nrow = min(5, omega_dim_current)
        self.viz.plotly_grid(
            plot_type='quiver',
            hmap_batch=pi_opt_grid,
            ncols=nrow,
            key="off_policy_pi",
            action_space_type=self.action_space_type,
            actions=self.vis_env.actions,
            subplot_titles=None,
            normalize_mode=None,
            iter_id=iter_id,
            save_figures=True,
        )
        return info

    def get_hmap(self, pos, value=None, freq_normalize=True):
        if hasattr(pos, 'cpu'):
            pos = pos.cpu().numpy()

        _H = self.vis_env.height
        _W = self.vis_env.width
        pos = pos.astype('int')
        _x = pos[:, 0]
        _y = pos[:, 1]
        _fill_pos = np.zeros((pos.shape[0], _H, _W))
        _fill_pos[np.arange(pos.shape[0]), _x, _y] = 1
        if value is not None:
            if value.shape[1] != 1:
                _fill_val = np.zeros((pos.shape[0], _H, _W, value.shape[1]))
                _fill_val[np.arange(pos.shape[0]), _x, _y] = value[:, :]
                _normalizer = (_fill_pos.sum(
                    0, keepdims=True)[:, :, :, np.newaxis] + 1e-9)
            else:
                _fill_val = np.zeros((pos.shape[0], _H, _W))
                _fill_val[np.arange(pos.shape[0]), _x, _y] = value[:, 0]
                _normalizer = (_fill_pos.sum(0, keepdims=True) + 1e-9)

            if freq_normalize:
                # Normalization by pos XY frequency
                _fill_val = _fill_val / _normalizer
            return _fill_val.sum(0).T
        else:
            return _fill_pos.sum(0).T


class TrainLoggerTD(object):
    def __init__(self,
        args,
        vis_env,
    ):
        self.args = args
        self.vis_env = vis_env

        self.init_visdom_env()
        self.best_across_train = {}

    def init_visdom_env(self):
        self.args.output_log_path = os.path.join(
            self.args.log_dir, self.args.time_id + '_' + self.args.visdom_env_name + '.log')
        self.viz = VisdomLogger(env_name=self.args.visdom_env_name,
                                server=self.args.server,
                                port=self.args.port,
                                log_file=self.args.output_log_path,
                                fig_save_dir=self.args.save_dir,
                                win_prefix=self.args.identifier)
                                # win_prefix=self.args.time_id + '_' + self.args.identifier)
        print("Logging visdom update events to: {}".format(
            self.args.output_log_path))
        pprint.pprint(vars(self.args))
        self.viz.text(vars(self.args), "params")

        vis_obs, vis_info = self.vis_env.reset()
        assert 'rgb_grid' in vis_info
        self.env_rgb_img = vis_info['rgb_grid'].transpose([2, 0, 1])
        self.env_rgb_img = np.flip(self.env_rgb_img, 1)
        self.viz.image(self.env_rgb_img, 'env_rgb_img')

    def plot_success(
        self,
        prefix: str,
        total_num_steps,
        rewards,
        success,
        mrids,
        track_best=False,
    ):
        self.plot_quad_stats(
            x_val = total_num_steps,
            plot_title = prefix + "episode_reward",
            array = rewards,
        )
        self.plot_quad_stats(
            x_val = total_num_steps,
            plot_title = prefix + "mrids",
            array = mrids,
        )
        self.viz.line(total_num_steps, success.mean(), prefix + "success", "mean",
            xlabel="time_steps")

        best_success_achieved: bool = False
        if track_best:
            if prefix not in self.best_across_train.keys():
                self.best_across_train[prefix] = success.mean()
            else:
                if success.mean() > self.best_across_train[prefix]:
                    self.best_across_train[prefix] = success.mean()
                    best_success_achieved = True
                else:
                    pass
            self.viz.line(total_num_steps, self.best_across_train[prefix],
                prefix + "success", "best_mean", xlabel="time_steps")
        return best_success_achieved

    def plot_quad_stats(self, x_val, plot_title, array, xlabel="time_steps"):
        self.viz.line(x_val, np.mean(array),
            plot_title, "mean", xlabel=xlabel)
        self.viz.line(x_val, np.median(array),
            plot_title, "median", xlabel=xlabel)
        self.viz.line(x_val, np.max(array),
            plot_title, "max", xlabel=xlabel)
        self.viz.line(x_val, np.min(array),
            plot_title, "min", xlabel=xlabel)
