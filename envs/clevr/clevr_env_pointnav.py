import os
import numpy as np
import torch

from envs.clevr.controller import Controller
from envs.clevr.clevr_env_base import ClevrEnvBase
import metrics

from gym import spaces

def flatten_last_two_dims(tensor):
    return tensor.view(
        *tensor.shape[:-2], tensor.shape[-1]*tensor.shape[-2])

class ClevrEnvPointNav(ClevrEnvBase):
    """
    ClevrEnv with a navigation task to a fixed observable target.

    In variable naming, 'target' refers to the 'goal' object in
    the grid world environment.
    """

    def __init__(
            self,
            env_base_dir,
            pov_imgs_path,
            env_list,
            # Attrs that can be used as targets
            # If 'None', use all available attrs
            attrs=None,
            num_agents=1,
            num_head_turns=8,
            num_body_turns=4,
            target_type='one-hot',
            grid_size=5,
            f_cell_occ_map='cell_occ_map.npy',
            f_cell_attr_map='cell_attr_map.npy',
            env_dim=6,
            viz_dir='data/viz',
            # question_dir='data/questions',
            img_dim=64,
            enable_viz=False,
            observability='full',
            action_space_id=0,
            num_steps=50,
            reward_type='neg_l2',
            # encoder_checkpoint=False,
            attr_count=96,
            attr_dims=None,
            # Used for single target training
            target_limit=-1,
            # Used for validation
            same_target=False,
            spawn_curriculum='none',
            early_stopping=True,
            mask_actions_on_done=False,
            change_attrs_at_eval= True,
            max_distractors=2,
            reachability_thresholds=[2, 4],
            terminal_reward=0,
            spike_value=5,
            train_reachability_threshold=4,
            # Set to 'True' if using the env with abstract goals
            is_target_underspecified=False,
            # Mask specifying which attributes in the sampled goal to
            # reveal as observed.
            abs_target_mask=None,
            potential_type='l1'):

        super(ClevrEnvPointNav, self).__init__(
                env_base_dir=env_base_dir,
                pov_imgs_path=pov_imgs_path,
                env_list=env_list,
                num_agents=num_agents,
                num_head_turns=num_head_turns,
                num_body_turns=num_body_turns,
                grid_size=grid_size,
                f_cell_occ_map=f_cell_occ_map,
                f_cell_attr_map=f_cell_attr_map,
                env_dim=env_dim,
                viz_dir=viz_dir,
                # question_dir,
                img_dim=img_dim,
                action_space_id=action_space_id,
                spawn_curriculum=spawn_curriculum,
                observability=observability)

        assert observability in ['full'], \
                "Only 'full' observability supported for now."

        self.observability = observability
        self.target_type = target_type
        assert num_steps > 0, "Positive episode len required"
        self.num_steps = num_steps
        self.reward_type = reward_type
        self.attr_count = attr_count
        self.attr_dims = attr_dims


        if attrs is None:
            attrs = list(range(self.attr_count))

        self.abs_target_mask = abs_target_mask

        # These are the global attributes that can be chosen as targets.

        #[NOTE]: In case of Abstraction Validation/Test Experiments,
        # attrs will be the valid objects with respect to the partial
        # goal specification.
        self.attrs = attrs

        self.target_limit = target_limit
        self.same_target = same_target
        self.spawn_curriculum = spawn_curriculum
        self.early_stopping = early_stopping
        self.mask_actions_on_done = mask_actions_on_done

        self.reachability_thresholds = reachability_thresholds
        self.change_attrs_at_eval = change_attrs_at_eval

        self._SAMPLE_LIMIT = 10
        self.max_distractors = max_distractors

        self.action_space_id = action_space_id

        self.terminal_reward = terminal_reward
        self.spike_value = spike_value

        self.train_reachability_threshold = train_reachability_threshold

        # Abstraction
        self.is_target_underspecified = is_target_underspecified

        self.compute_recall = self.is_target_underspecified

        self.potential_type = potential_type

        if self.is_target_underspecified:
            assert abs_target_mask, "Mask needed in case of underspecified goals"

            self.abs_target_mask = np.array(abs_target_mask)
        else:
            self.abs_target_mask = np.ones(4,int)


        if self.action_space_id == 0:
            '''
            Action space is movement in four cardinal directions
            N, S, E, W and a STOP action to end the episode
            '''
            self.action_space = spaces.Discrete(5)
        else:
            raise NotImplementedError

        if self.observability == 'full':
            # [NOTE] : Really make sure that this is correct
            shape = (self.num_agents,
                # HxW + agent_orientation + target_xy
                (self.grid_size * self.grid_size * \
                sum(self.attr_dims)) + 4 + 2)

            low = 0.0
            high = float(self.grid_size)

            self.observation_space = spaces.Box(
                    low=low, high=high, shape=shape, dtype=np.float32)

        else:
            raise NotImplementedError

    def reset(self):
        super(ClevrEnvPointNav, self).reset()

        self._task_reset()

        obs = self._get_agent_obs()
        return obs

    def step(self, actions):
        if self.mask_actions_on_done and self._step > 0:
            if self.action_space_id == 0:
                actions[np.where(self._done[self._step - 1] == 1)] = 0
            else:
                raise NotImplementedError

        self._actions_taken[self._step] = actions
        # return super().step(actions)

        super(ClevrEnvPointNav, self).step(actions)

        obs = self._get_agent_obs()
        # '''NOTE: _is_done MUST be called before _compute_reward'''
        done = self._is_done()
        reward = self._compute_reward()
        info = self._get_info()

        self._step += 1

        return obs, reward, done, info

    def load_viewpoints(self, loaded_viewpoints):
        # self._ctrl._load_obs(loaded_viewpoints)
        self.loaded_viewpoints = loaded_viewpoints #.numpy()

    def _task_reset(self, manual_targets=None):
        """
        Reset the Environment with new agent configs, clevr environment,
        and a target location.
        """

        # Copy the cell_attr_map for all the agents
        self.per_agent_cell_attr_map = np.tile(
                self.cell_attr_map[np.newaxis, :],
                (self.num_agents, 1, 1, 1, 1, 1, 1))

        # Reset action storage (required for evaluating navigation)
        self._actions_taken = -1 * np.ones(
            (self.num_steps, self.num_agents), dtype='int')

        # Reset done tensor

        # Index of the current step
        self._step = 0

        # self._done = self._has_stopped || self._is_episode_done
        self._done = np.zeros((self.num_steps, self.num_agents), dtype='int')
        self._has_stopped = np.zeros((self.num_steps, self.num_agents), dtype='int')
        self._is_episode_done = np.zeros((self.num_agents), dtype='int')

        # Reload the object list in the environment.
        self._obj_map = self._objects_from_attr_map()
        self._obj_list = sorted(list(self._obj_map.keys()))
        self._obj_id_to_pos = {obj:pos for pos, obj in enumerate(self._obj_list)}

        _obj_xy = []
        _obj_attrs = []
        for obj in self._obj_list:
            obj_x = self._obj_map[obj]['x_mean']
            obj_y = self._obj_map[obj]['y_mean']
            obj_attr = self._obj_map[obj]['attr']
            _obj_xy.append(np.array([obj_x, obj_y]))
            _obj_attrs.append(obj_attr)
        self._obj_xy = np.stack(_obj_xy, 1)
        self._obj_attrs = np.stack(_obj_attrs, 1)

        self._avail_targets = list(set(self._obj_list) & set(self.attrs))

        # Set targets
        self._init_targets(manual_targets=manual_targets)

        self._update_cmap_attrs()

    def _init_targets(self, manual_targets=None):
        # Point Navigation requires a target; Sample a new one for each agent

        assert len(self._avail_targets) > 0

        if self.target_limit <= 0:
            # Use all the objects
            available_target_list = self._avail_targets
        else:
            available_target_list = self._avail_targets[:self.target_limit]

        if manual_targets is not None:
            import pdb; pdb.set_trace()

        elif self.same_target:
            single_target_idx = np.random.choice(available_target_list, 1)
            target_idx = np.repeat(
                single_target_idx, self.num_agents, axis=0)

        else: # Sample from available targets
            target_idx = np.random.choice(
                available_target_list, size=self.num_agents)

        target_xy = self._get_goal_xy(target_idx)
        self.targets = (target_idx,target_xy)

        # Initializing for dense L2 reward
        self._initial_dist_l2 = self._compute_distance_from_goal()
        self._prev_l2_dist = self._initial_dist_l2

        self._initial_dist_l1 = self._compute_distance_from_goal(ord=1)
        self._prev_l1_dist = self._initial_dist_l1

        if self.reward_type == 'dense_spiky_l2':
            self._initial_dist_spiky_l2 = \
                self._compute_distance_from_goal(ord=2, spiky=True)
            self._prev_dist_spiky_l2 = self._initial_dist_spiky_l2


    def _get_goal_xy(self, targets):
        """
        Get mean x and y coordinates of target objects
        """

        target_x = np.array([self._obj_map[obj]['x_mean'] \
                for obj in targets])

        target_y = np.array([self._obj_map[obj]['y_mean'] \
                for obj in targets])

        target_xy = np.stack((target_x,target_y),1)

        return target_xy

    def _get_agent_obs(self):
        """
        Fetch agent observation.

        'full' : Fully specified current state of the environment
            in the form of an occupancy grid with object attributes
            and agent's current XY position.

        'partial': Currently not supported.
        """

        if self.observability == 'full':
            # A*30*30*96 tensor
            env_obs = self._get_full_obs()
        else:
            raise NotImplementedError

        # A*4 tensor where dim1 is (X, Y, body_pos, body_pos)
        # NOTE: body_pos is obsolete for cardinal action space
        # i.e. XY alone is the complete agent config
        agent_cfg = self.get_agent_cfg()

        if self.observability == 'full':
            cmap = env_obs
            agent_x = agent_cfg[:, 0]
            agent_y = agent_cfg[:, 1]
            # targets = np.array([ob[0] for ob in obs])
            target_ids = self.targets[0]

            if self.target_type == 'one-hot':
                # [NOTE] Not entirely sure if this should be supported
                raise NotImplementedError
                targets = target_ids.reshape(target_ids.shape[0],1)

            elif self.target_type == 'k-hot':
                target_attrs = []
                for t_id in target_ids:
                    orig_attr = self._obj_map[t_id]['attr'].copy()
                    # [NOTE] : Set all unspecified attrs to -1
                    orig_attr[self.abs_target_mask == 0] = -1
                    target_attrs.append(orig_attr)

                target_attrs = np.stack(target_attrs, 0)
                targets = target_attrs

                attrs_shape = self.cell_attr_map.shape[2:]
                # attrs_shape = (3, 8, 2, 2)
                cmap = cmap.reshape(*cmap.shape[:3], *attrs_shape)
                # cmap is (B, 30, 30, 3, 8, 2, 2)
                cmap_k_hot = []
                # cmap_k_hot will be (B, 30, 30, 4)
                for attr_index in range(len(attrs_shape)):
                    _axes = [3, 4, 5, 6]
                    del _axes[attr_index]
                    # cmap_k_hot.append(
                    #     cmap.argmax(3 + attr_index).sum((3,4,5)))
                    cmap_k_hot.append(cmap.sum(tuple(_axes)))
                # cmap_k_hot = np.stack(cmap_k_hot, 1)
                cmap_k_hot = np.concatenate(cmap_k_hot, -1)
                cmap = cmap_k_hot.transpose(0, 3, 1, 2)

            agent_info = 1.0 * agent_cfg[:, :2]

            # A*15*30*30 : 15 is k-hot vector for describing the
            # attribute of object
            cmap = 1.0 * cmap
            # A*4 tensor
            targets = 1.0 * targets

            # A*2
            agent_info = agent_info / self.grid_size

            obs = np.concatenate(
                [targets, agent_info, cmap.reshape(cmap.shape[0], -1)], 1)

            # obs = obs.astype('int')
            obs = obs.astype('float')
            obs = torch.from_numpy(obs)

        else:
            raise NotImplementedError

        return obs

    def get_cell_occ_map(self):
        """
        [Note]: Super hacky

        Gives the top down view of the environment. This is also the observation
        in case of full observability. So ideally should be easier to share
        this computation. But that's a story for better days.
        """
        cmap = self._get_full_obs()
        return cmap

    def get_processed_pov_img(self):
        """
        [Note]: Hacky function used for visualization

        Returns a numpy array
        """

        imgs = self._ctrl._fetch_obs()

        if self.observability == 'partial-triplet':
            imgs = np.transpose(imgs,(0,1,4,2,3))
            _shape = imgs.shape
            _new_shape = [_shape[0], _shape[1] * _shape[2], *_shape[3:]]
            imgs = np.reshape(imgs, _new_shape)

        else:
            raise NotImplementedError

        return imgs

    def _compute_reward(self):
        """
        Select which reward to use.

        'give_reward' determines the time step when a reward is due. It is first
        set at the time step when an episode is done, and then unset for the
        subsequent time steps.
        """

        if self.reward_type == 'neg_l2':
            reward_fn = self._neg_l2_reward

        elif self.reward_type == 'neg_l1':
            reward_fn = self._neg_l1_reward

        elif self.reward_type == 'dense_l2':
            reward_fn = self._dense_l2_reward

        elif self.reward_type == 'dense_l1':
            reward_fn = self._dense_l1_reward

        elif self.reward_type == 'dense_spiky_l2':
            reward_fn = self._dense_spiky_l2_reward

        elif self.reward_type == 'sparse_spiky_l2':
            reward_fn = self._sparse_spiky_l2_reward

        elif self.reward_type == 'neg_dense_l2':
            reward_fn = self._neg_dense_l2_reward

        elif self.reward_type == 'dense_l1_xpe':
            reward_fn = self._dense_l1_xpe_reward

        elif self.reward_type == 'target_reached':
            reward_fn = self._target_reached_reward

        elif self.reward_type == 'feat_sim':
            reward_fn = self._feat_sim_reward

        elif self.reward_type == 'exp_l2':
            reward_fn = self._dense_exp_l2_reward

        elif self.reward_type == 'dense_binary_v1':
            reward_fn = self._dense_binary_l2_reward_v1

        elif self.reward_type == 'dense_binary_v2':
            reward_fn = self._dense_binary_l2_reward_v2

        elif self.reward_type == 'pot_diff_reshaped':
            reward_fn = self._pot_diff_reshaped
        else:
            raise ValueError("Invalid reward type '{0}' specified."\
                .format(reward_type))

        if self._step == 0:
            give_reward = self._done[self._step]
        else:
            # Reward is given the first time done flag is triggered
            if self.reward_type == 'sparse_spiky_l2':
                give_reward = self._has_stopped[self._step] * \
                              np.logical_not(self._has_stopped[self._step-1])
            else:
                give_reward = self._done[self._step] * \
                              np.logical_not(self._done[self._step-1])

        # [NOTE] : Not using give_reward at all now
        #if self._step == 0:
        #    give_reward = self._has_stopped[0]
        #else:
        #    give_reward = self._has_stopped[self._step] * \
        #            np.logical_not(self._has_stopped[self._step-1])

        return reward_fn(give_reward)

    def _sparse_spiky_l2_reward(self, give_reward):
        # if self._step == 0:
        #     give_terminal_reward = self._has_stopped[0]
        # else:
        #     give_terminal_reward = np.logical_and(self._has_stopped[self._step],
        #             np.logical_not(self._has_stopped[self._step-1]))

        if give_reward.sum() == 0:
            # Do not compute reward if no reward is to be given
            return give_reward.astype('float')

        # Normalize
        NORMALIZER_L2 = np.sqrt(2 * self.grid_size * self.grid_size)

        dist_l2 = self._compute_distance_from_goal(ord=2, spiky=True)
        MAX_DIST = NORMALIZER_L2

        reward = NORMALIZER_L2 - dist_l2

        reward /= NORMALIZER_L2
        reward = reward * (give_reward.astype('float'))

        # has_reached = self._has_reached()
        #
        # reward += self.terminal_reward * has_reached.astype(float) * \
        #                 give_terminal_reward.astype(float)

        return reward

    def _neg_l2_reward(self, give_reward):
        """
        Normalized Negative L2 distance of the agent from the target.

        This is a sparse reward.
        """
        raise NotImplementedError

        # if self._step == 0:
        #     give_terminal_reward = self._has_stopped[0]
        # else:
        #     give_terminal_reward = np.logical_and(self._has_stopped[self._step],
        #             np.logical_not(self._has_stopped[self._step-1]))

        if give_reward.sum() == 0:
            # Do not compute reward if no reward is to be given
            return give_reward.astype('float')

        dist_l2 = self._compute_distance_from_goal()
        reward = -1*dist_l2

        # Normalize
        NORMALIZER_L2 = np.sqrt(2 * self.grid_size * self.grid_size)

        reward /= NORMALIZER_L2
        reward = reward * (give_reward.astype('float'))

        # has_reached = self._has_reached()
        #
        # reward += self.terminal_reward * has_reached.astype(float) * \
        #                 give_terminal_reward.astype(float)

        return reward

    def _neg_l1_reward(self, give_reward):
        '''
        Normalized Negative L1 distance of the agent from the target.
        '''
        raise NotImplementedError

        # if self._step == 0:
        #     give_terminal_reward = self._has_stopped[0]
        # else:
        #     give_terminal_reward = np.logical_and(self._has_stopped[self._step],
        #             np.logical_not(self._has_stopped[self._step-1]))

        if give_reward.sum() == 0:
            # do not compute reward if no reward is to be given
            return give_reward.astype('float')

        dist_l1 = self._compute_distance_from_goal(ord=1)
        reward = -1*dist_l1

        # normalize
        normalizer_l2 = np.sqrt(2 * self.grid_size * self.grid_size)

        reward /= normalizer_l2
        reward = reward * (give_reward.astype('float'))

        # has_reached = self._has_reached()
        #
        # reward += self.terminal_reward * has_reached.astype(float) * \
        #                 give_terminal_reward.astype(float)
        return reward

    def _dense_spiky_l2_reward(self, give_reward, normalized=True):
        if self._step == 0:
            give_terminal_reward = self._has_stopped[0]
        else:
            give_terminal_reward = np.logical_and(self._has_stopped[self._step],
                    np.logical_not(self._has_stopped[self._step-1]))

        curr_dist = self._compute_distance_from_goal(ord=2, spiky=True)
        improvement = self._prev_l2_dist - curr_dist
        self._prev_l2_dist = curr_dist.copy()

        # [NOTE] : Do we want to try some other normalization for this reward.
        if normalized:
            NORMALIZER_L2 = np.sqrt(2 * self.grid_size * self.grid_size)
            improvement = improvement / NORMALIZER_L2

        intermediate_reward = improvement * \
                np.logical_not(self._done[self._step]).astype(float)

        has_reached = self._has_reached()

        give_terminal_reward = give_terminal_reward.astype(float) * has_reached.astype(float)

        # Add terminal reward component
        terminal_reward = self.terminal_reward * give_terminal_reward

        reward = intermediate_reward + terminal_reward

        return reward

    def _dense_l2_reward(self, give_reward, normalized=True):
        """
        Normalized decrease in the L2 distance of the agent to the target.

        This is a dense reward.

        Normalization constant is \sqrt(2*N*N), where N is the grid size.
        This reward is not gauranteed to be negative. So can't be used for
        max_ent objective.

        """

        # Recompute 'give_reward' for this specific reward

        if self._step == 0:
            give_terminal_reward = self._has_stopped[0]
        else:
            give_terminal_reward = np.logical_and(self._has_stopped[self._step],
                    np.logical_not(self._has_stopped[self._step-1]))

        curr_dist = self._compute_distance_from_goal()
        improvement = self._prev_l2_dist - curr_dist
        self._prev_l2_dist = curr_dist.copy()

        # [NOTE] : Do we want to try some other normalization for this reward.
        if normalized:
            NORMALIZER_L2 = np.sqrt(2 * self.grid_size * self.grid_size)
            improvement = improvement / NORMALIZER_L2

        intermediate_reward = improvement * \
                np.logical_not(self._done[self._step]).astype(float)

        has_reached = self._has_reached()

        if self.terminal_reward > 0 :
            give_terminal_reward = give_terminal_reward.astype(float) * has_reached.astype(float)

            # Add terminal reward component
            terminal_reward = self.terminal_reward * give_terminal_reward
        else:
            terminal_reward = 0.0

        reward = intermediate_reward + terminal_reward

        return reward

    def _dense_l1_xpe_reward(self, give_reward, normalized=True):
        if self._step == 0:
            give_terminal_reward = self._has_stopped[0]
        else:
            give_terminal_reward = np.logical_and(self._has_stopped[self._step],
                    np.logical_not(self._has_stopped[self._step-1]))

        curr_dist = self._compute_distance_from_goal(ord=1)
        EXISTENCE_PENALTY = 0.005
        improvement = self._prev_l1_dist - curr_dist - EXISTENCE_PENALTY
        self._prev_l1_dist = curr_dist.copy()

        # [NOTE] : Do we want to try some other normalization for this reward.
        if normalized:
            NORMALIZER_L2 = np.sqrt(2 * self.grid_size * self.grid_size)
            reward = improvement / NORMALIZER_L2
        else:
            reward = improvement

        reward = reward * np.logical_not(self._done[self._step]).astype(float)

        has_reached = self._has_reached()

        give_terminal_reward = give_terminal_reward.astype(float) * has_reached.astype(float)

        # Add terminal reward component
        reward += self.terminal_reward * give_terminal_reward.astype(float)

        return reward

    def _dense_l1_reward(self, give_reward, normalized=True):
        """
        Normalized decrease in the L1 distance of the agent to the target.

        This is a dense reward.

        Normalization constant is \sqrt(2*N*N), where N is the grid size.
        This reward is not gauranteed to be negative. So can't be used for
        max_ent objective.

        """

        if self._step == 0:
            give_terminal_reward = self._has_stopped[0]
        else:
            give_terminal_reward = np.logical_and(self._has_stopped[self._step],
                    np.logical_not(self._has_stopped[self._step-1]))

        curr_dist = self._compute_distance_from_goal(ord=1)
        improvement = self._prev_l1_dist - curr_dist
        self._prev_l1_dist = curr_dist.copy()

        # [NOTE] : Do we want to try some other normalization for this reward.
        if normalized:
            NORMALIZER_L2 = np.sqrt(2 * self.grid_size * self.grid_size)
            reward = improvement / NORMALIZER_L2
        else:
            reward = improvement

        reward = reward * np.logical_not(self._done[self._step]).astype(float)

        has_reached = self._has_reached()

        give_terminal_reward = give_terminal_reward.astype(float) * has_reached.astype(float)

        # Add terminal reward component
        reward += self.terminal_reward * give_terminal_reward.astype(float)

        return reward

    def _dense_exp_l2_reward(self, give_reward):
        """
        r_t = exp(-0.75*d_t), d_t = distance from target at time t.
        """

        dist_l2 = self._compute_distance_from_goal()
        reward = np.exp(-0.75*dist_l2)

        reward = reward * np.logical_not(self._done[self._step])

        return reward

    def _neg_dense_l2_reward(self, give_reward):
        """
        Shifted Normalized decrease in the distance of the agent to the target
        (shifted to make it negative).

        This is a dense reward.

        There is an extra reward at the end of the episode. This equals
        negative of the number of steps taken by the agent, normalized by maximum
        length of the episode.

        End of an episode is marked when, either
        1) Max Episode Length is elapsed, or
        2) agent chooses to end the episode by emitting an END TOKEN)
        """
        # give_reward is ignored
        curr_dist = self._compute_distance_from_goal()
        improvement = self._prev_l2_dist - curr_dist
        self._prev_l2_dist = curr_dist.copy()

        NORMALIZER_L2 = np.sqrt(2 * self.grid_size * self.grid_size)

        end_reward  = -1 * (self._step+1) / self.num_steps
        reward = (improvement - 1) / NORMALIZER_L2

        reward = reward * np.logical_not(self._done[self._step])

        reward = reward + (give_reward * end_reward)
        assert reward.max() < 1e-6, "Reward should be negative!"
        return reward

    def _neg_l1_reshaped(self, give_reward):
        '''
        reward = terminal * stopped * has_reached
                 + failed_stop_penalty * stopped * not_reached
                 + not_stopped_penalty * not_stopped * episode_ended
                 + staying_penalty * not_stopped * episode_not_ended
        '''

        normalizer_l1 = 2 * self.grid_size

        dist_l1 = self._compute_distance_from_goal(ord=1)
        dist_l1 /= normalizer_l1

        # We need rewards, not costs
        neg_l1 = -1 * dist_l1

        reached = self.has_reached().astype(float)
        not_reached = np.logical_not(reached)
        reached, not_reached = reached.astype(float), not_reached.astype(float)

        stopped = self._has_stopped[self._step]
        not_stopped = np.logical_not(stopped)
        stopped, not_stopped = stopped.astype(float), not_stopped.astype(float)

        episode_ended = self._is_episode_done
        episode_not_ended = np.logical_not(episode_ended)
        episode_ended, episode_not_ended = episode_ended.astype(float), \
                                            episode_non_ended.astype(float)

        self.failed_stop_penalty = neg_l1
        self.not_stopped_penalty = neg_l1

        reward = self.terminal_reward * stopped * reached \
                + self.failed_stop_penalty * stopped * not_reached \
                + self.not_stopped_penalty * not_stopped * episode_ended

        return reward

    def _neg_l2_reshaped(self, give_reward):
        '''
        reward = terminal * stopped * has_reached
                 + failed_stop_penalty * stopped * not_reached
                 + not_stopped_penalty * not_stopped

        For this case, failed_stop_penalty = neg_l2
        '''

        normalizer_l2 = np.sqrt(2 * self.grid_size * self.grid_size)

        dist_l2 = self._compute_distance_from_goal(ord=2)
        dist_l2 /= normalizer_l2

        # We need rewards, not costs
        neg_l2 = -1 * dist_l2

        reached = self.has_reached().astype(float)
        not_reached = np.logical_not(reached)
        reached, not_reached = reached.astype(float), not_reached.astype(float)

        stopped = self._has_stopped[self._step]
        not_stopped = np.logical_not(stopped)
        stopped, not_stopped = stopped.astype(float), not_stopped.astype(float)

        episode_ended = self._is_episode_done
        episode_not_ended = np.logical_not(episode_ended)
        episode_ended, episode_not_ended = episode_ended.astype(float), \
                                            episode_non_ended.astype(float)

        self.failed_stop_penalty = neg_l2
        self.not_stopped_penalty = neg_l2

        reward = self.terminal_reward * stopped * reached \
                + self.failed_stop_penalty * stopped * not_reached \
                + self.not_stopped_penalty * not_stopped * episode_ended

        return reward

    def _pot_diff_reshaped(self, give_reward):
        '''
        reward = terminal * stopped * reached
                + delta * not_stopped
                + failed_stop_penalty * stopped * not_reached
        '''

        if self.potential_type == 'l1':

            curr_dist = self._compute_distance_from_goal(ord=1)
            improvement = self._prev_l1_dist - curr_dist

            self._prev_l1_dist = curr_dist.copy()

            NORMALIZER_L1 = 2 * self.grid_size
            delta = improvement / NORMALIZER_L1

        elif self.potential_type == 'l2':

            curr_dist = self._compute_distance_from_goal(ord=2)
            improvement = self._prev_l2_dist - curr_dist

            self._prev_l2_dist = curr_dist.copy()

            NORMALIZER_L2 = np.sqrt(2 * self.grid_size * self.grid_size)
            delta = improvement / NORMALIZER_L2

        reached = self._has_reached().astype(float)
        not_reached = np.logical_not(reached)
        reached, not_reached = reached.astype(float), not_reached.astype(float)

        if self._step == 0:
            stopped = self._has_stopped[0]
        else:
            stopped = np.logical_and(self._has_stopped[self._step],
                        np.logical_not(self._has_stopped[self._step-1]))

        not_stopped = np.logical_not(stopped)
        stopped, not_stopped = stopped.astype(float), not_stopped.astype(float)

        # Set this to distance from the target
        if self.potential_type == 'l1':
            self.failed_stop_penalty = -1 * self._prev_l1_dist / NORMALIZER_L1
        elif self.potential_type == 'l2':
            self.failed_stop_penalty = -1 * self._prev_l2_dist / NORMALIZER_L2

        reward = self.terminal_reward * stopped * reached \
                + self.failed_stop_penalty * stopped * not_reached \
                + delta * not_stopped

        # Add terminal reward component
        reward += self.terminal_reward * give_reward.astype(float)

        return reward

    def _target_reached_reward(self, give_reward):
        """
        A sparse reward, given at the end of the episode, if the target
        is reached.

        Target is reached if at least one cell which the target occupies is
        within the 3x3 grid centered at the agent's final cell.

        [NOTE] : FirstName can elaborate on the idea for normalization.

        Small negative reward (~ -0.1) for not reaching object.
        Big positive reward (~ +0.9) for reaching object.
        Normalization is based on number of cells occupied by objects.
        """

        overlap_check = self._has_reached()
        overlap_check = overlap_check.astype('int')


        occ_cell_count = self.cell_occ_map.sum()
        norm = 1.5 / (self.grid_size * self.grid_size)
        offset = norm * occ_cell_count
        reward = overlap_check - offset

        reward = reward * (give_reward.astype('float'))
        return reward

    def _dense_binary_l2_reward_v1(self, give_reward):
        '''
        A dense reward with a value:
        +1 : If current step reduces the distance to target
        0  : Otherwise

        A terminal reward is also given if agent successfully navigates to the
        goal within the specified threshold 'self.train_reachability_threshold'
        '''

        # give_reward is ignored
        curr_dist = self._compute_distance_from_goal()
        improvement = self._prev_l2_dist - curr_dist
        self._prev_l2_dist = curr_dist.copy()
        reward =  (improvement > 0).astype(float)

        reward = reward * np.logical_not(self._done[self._step])

        reached = self._has_reached().astype(float)

        terminal_reward = self.terminal_reward * \
                reached * give_reward.astype(float)

        reward += terminal_reward

        return reward

    def _dense_binary_l2_reward_v2(self, give_reward):
        '''
        A dense reward with a value:
        +1 : If current step reduces the distance to target
        -1 : Otherwise

        A terminal reward is also given if agent successfully navigates to the
        goal within the specified threshold 'self.train_reachability_threshold'
        '''

        # give_reward is ignored
        curr_dist = self._compute_distance_from_goal()
        improvement = self._prev_l2_dist - curr_dist
        self._prev_l2_dist = curr_dist.copy()


        reward_pos =  (improvement > 0).astype(float)
        reward_neg = -1*(improvement <= 0).astype(float)

        reward = reward_pos + reward_neg

        reward = reward * np.logical_not(self._done[self._step])

        reached = self._has_reached().astype(float)

        terminal_reward = self.terminal_reward * \
                reached * give_reward.astype(float)

        reward += terminal_reward

        return reward

    def _feat_sim_reward(self, give_reward):
        """
        Not used anymore.
        """
        raise NotImplementedError
        # '''UNCOMMENT after debugging'''
        if give_reward.sum() == 0:
            # Do not compute reward if no reward is to be given
            return give_reward.astype('float')
        # '''UNCOMMENT after debugging'''

        # Chosen from observing norm differences in data
        FEAT_SIM_NORMALIZER = 10.0
        MAX_NORM = 8.0

        reward = []
        for id, flag in enumerate(give_reward):
            # '''REMOVE after debugging'''
            # flag = True
            # '''REMOVE after debugging'''
            if flag:
                agent_pov_img = self._ctrl._fetch_obs(id)
                center_img = agent_pov_img[1]
                img_feat = self._encode_img(center_img)

                target_id = self.targets[0][id]

                diff = img_feat - self._goal_views[target_id]

                norm_val = np.linalg.norm(diff, ord=2, axis=1)
                norm_val = -1 * norm_val.min()
                norm_val = np.clip(norm_val, -MAX_NORM, 0)
                norm_val = norm_val / FEAT_SIM_NORMALIZER
                reward.append(norm_val)
            else:
                reward.append(0.0)

        return np.array(reward)

    def _has_reached(self):
        '''
        Returns A*1 array of boolean values representing whether an agent has
        reached the target within 'self.train_reachability_threshold'.
        '''

        object_map = self._obj_map
        agent_cfg = self.get_agent_cfg()
        target_obj = [t for t in self.targets[0]]
        tr = lambda key: np.array([object_map[id][key] for id in target_obj])

        agent_x_max = (agent_cfg[:, 0] + 1).clip(0, 30)
        agent_x_min = (agent_cfg[:, 0] - 1).clip(0, 30)
        agent_y_max = (agent_cfg[:, 1] + 1).clip(0, 30)
        agent_y_min = (agent_cfg[:, 1] - 1).clip(0, 30)
        object_x_max = tr('x_max')
        object_x_min = tr('x_min')
        object_y_max = tr('y_max')
        object_y_min = tr('y_min')

        th = self.train_reachability_threshold

        x_check = np.logical_or(agent_x_max < (object_x_min - th),
                                agent_x_min > (object_x_max + th))

        y_check = np.logical_or(agent_y_max < (object_y_min - th),
                                agent_y_min > (object_y_max + th))

        overlap_check = np.logical_not(np.logical_or(x_check, y_check))

        return overlap_check

    def _is_done(self):
        """
        Keeps track of agents' progress for the current episode.

        Following member variables are computed:

        self._is_episode_done : True if self._step == self.num_steps -1, False
                                otherwise.

        self._has_stopped : (T,A) boolean np array. Once the agent has stopped,
                            this flag is always set till the end of the episode
                            for that agent.

        self._done        : self._is_episode_doen || self._has_stopped

        """

        if self.action_space_id == 0:
            stop_action = 0
        else:
                raise NotImplementedError

        self._is_episode_done = (self._step == (self.num_steps - 1) )

        #if self.early_stopping:
        #    _curr_done = np.logical_or(has_stopped, final_step_reached)
        #else:
        #    raise NotImplementedError
        #    _curr_done = final_step_reached

        has_stopped = np.array([actn == stop_action \
                               for actn in self.curr_actions])

        if self._step == 0:
            self._has_stopped[0] = has_stopped
        else:
            self._has_stopped[self._step] = np.logical_or(
                has_stopped, self._has_stopped[self._step-1])

        self._done[self._step] = np.logical_or(self._has_stopped[self._step],
                self._is_episode_done)

        return False

    def _get_info(self):
        """
        Additional information.
        """

        # Compute the current l2 distance
        current_dist = self._compute_distance_from_goal()

        # dist = np.stack((current_dist,self._initial_dist_l2),0)
        dist = current_dist

        done = self._done[self._step]

        info = {
            "initial_dist_l2": self._initial_dist_l2,
            "dist_l2": dist,
            "done": done,
        }

        if self._step == self.num_steps - 1:
            s_indicator, spl_values = self.evaluate_target_reached()
            info.update({
                "s_indicator" : s_indicator,
                "spl_values" : spl_values
            })

            if self.compute_recall:
                recall_counts = self.evaluate_recall()
                info.update({
                    'recall_counts' : recall_counts
                })

        return info

    def get_obj_attrs(self):
        all_attrs = np.array([self._obj_map[idx]['attr'] \
            for idx in self._obj_list])
        return all_attrs

    def get_tar_attrs(self):
        target_attrs = np.array([self._obj_map[idx]['attr'] \
            for idx in self.targets[0]])
        return target_attrs

    def get_nns(self, agent_cutoff):
        agent_cfg = self.get_agent_cfg()
        agent_cutoff = int(min(agent_cutoff, agent_cfg.shape[0]))
        agent_xy = agent_cfg[:agent_cutoff, :2]
        # obj_pos = np.array([self._obj_id_to_pos[obj] for obj in self._obj_map])

        agent_xy = agent_xy[:, :, np.newaxis]
        obj_xy = self._obj_xy[np.newaxis, :, :]

        delta = np.sqrt(((agent_xy - obj_xy)**2).sum(1))

        nn_object = delta.argmin(1)
        dist_l2 = delta.min(1)
        count_array = np.zeros((len(self._obj_list)))

        nn_flag = (dist_l2 < 4.0).astype('float')
        for idx, obj in enumerate(nn_object):
            count_array[obj] += 1 * nn_flag[idx]

        # all_attrs = np.array([self._obj_map[idx]['attr'] \
        #     for idx in self._obj_list])

        # return self._obj_attrs[:, nn_object], count_array, \
        #     target_attrs, all_attrs
        return count_array

    def evaluate_recall(self, bounding_box=True):
        """
        Returns np array of size OxT, with [o,t]  entry representing the count
        of agents reaching the goal object 'o' with 't' as the reachability
        threshold.
        """

        tr_bound = lambda key : \
                np.array([[self._obj_map[target][key]] for target in self.attrs])


        threshold = np.array(self.reachability_thresholds)

        target_min_x = tr_bound('x_min') - threshold
        target_min_y = tr_bound('y_min') - threshold
        target_max_x = tr_bound('x_max') + threshold
        target_max_y = tr_bound('y_max') + threshold



        agent_cfg = self.get_agent_cfg()

        agent_x = agent_cfg[:,[0], np.newaxis]
        agent_y = agent_cfg[:,[1], np.newaxis]

        x_check = np.logical_and(agent_x >= target_min_x,
                                 agent_x <= target_max_x)

        y_check = np.logical_and(agent_y >= target_min_y,
                                 agent_y <= target_max_y)

        succ_count = np.logical_and(x_check, y_check).astype(int)

        return np.sum(succ_count,axis=0)

    def evaluate_target_reached(self, bounding_box=True):
        """
        Arguments:
            bounding_box: If True, consider a bounding_box around the object
            as the object boundry. This is used to check if an agent has
            reached the target object.
        """
        agent_cfg = self.get_agent_cfg()
        # target_obj = [t for t in self.targets[0]]
        target_obj = self.targets[0]

        # MAX = self.grid_size

        success_values = []
        spl_values = []
        # path_ratios = []

        tr = lambda key: np.array(
            [self._obj_map[id][key] for id in target_obj])
        object_center_x = tr('x_mean')
        object_center_y = tr('y_mean')
        object_x_max = tr('x_max')
        object_y_max = tr('y_max')
        object_x_min = tr('x_min')
        object_y_min = tr('y_min')

        object_radius = np.sqrt(
            (object_center_x - object_x_max) ** 2 +\
            (object_center_y - object_y_max) ** 2
        )

        if bounding_box:
            measure_success_fn = metrics.measure_success_bbox
            success_args = (agent_cfg[:, 0], agent_cfg[:, 1],
                    object_x_min, object_y_min,
                    object_x_max, object_y_max)
        else:
            measure_success_fn = metrics.measure_success
            success_args = (agent_cfg[:, 0], agent_cfg[:, 1],
                object_center_x, object_center_y,
                object_radius)

        for threshold in self.reachability_thresholds:
            s_args = success_args + (threshold,)
            success_value = measure_success_fn(*s_args)
            success_values.append(success_value)

            agent_step_count = (self._actions_taken != 0).sum(0)

            #[NOTE] : trying out a more accurate way to measure distance
            #shortest_path_length = np.maximum(
            #    self._initial_dist_l1 - threshold, 1)

            side_len = np.minimum(tr('x_max') - tr('x_min'), tr('y_max') - tr('y_min'))

            corrected_initial_dist = self._initial_dist_l1 - \
                    (side_len / 2) - threshold

            shortest_path_length = np.maximum(corrected_initial_dist, 1)


            spl_value = metrics.measure_spl(success_value,
                                            agent_step_count,
                                            shortest_path_length)

            spl_values.append(spl_value)

        success_values = np.stack(success_values, 0)
        spl_values = np.stack(spl_values, 0)

        return success_values, spl_values

    def _objects_sorted_l2(self, reference_xy):
        # reference_xy has shape (*, 2)
        ref_x = reference_xy[:, 0:1]
        ref_y = reference_xy[:, 1:2]

        del_x = ref_x - self._obj_xy[0:1]
        del_y = ref_y - self._obj_xy[1:2]

        dist_vec = np.stack([del_x, del_y], 0)
        obj_dists = np.linalg.norm(dist_vec, axis=0)

        ordering = np.argsort(obj_dists, axis=1)
        # ordering shape is (*, num_objects)
        return ordering

    def _compute_distance_from_goal(self, ord=2, spiky=False):
        """
        Compute agent to goal distance.
        """

        agent_cfg = self.get_agent_cfg()

        target_xy = self.targets[1]

        x_dist = agent_cfg[:,0] - target_xy[:,0]
        y_dist = agent_cfg[:,1] - target_xy[:,1]

        dist_vec = np.array([x_dist, y_dist])
        dist = np.linalg.norm(dist_vec, ord=ord, axis=0)

        if spiky:
            has_reached = self._has_reached().astype('float')
            dist = dist - (self.spike_value * has_reached)

        return dist

    def _objects_from_attr_map(self):
        '''
        Get unique objects and the grid cells they occupy
        '''
        cell_attr = self.cell_attr_map
        H, W = cell_attr.shape[:2]
        cell_attr_flat = self.cell_attr_map.reshape(H, W, -1)

        object_map = {}
        for ids in zip(*np.where(cell_attr == 1)):
            attr = ids[2:]
            # attr_int = self._attr_to_int_map(attr)
            x_, y_ = ids[:2]
            attr_int = np.where(cell_attr_flat[x_, y_] == 1)[0][0]

            if attr_int not in object_map.keys():
                object_map[attr_int] = {
                    'attr': np.array(attr),
                    'locs': [ids[:2]],
                }
            else:
                object_map[attr_int]['locs'].append(ids[:2])

        # Compute mean object locations
        for key in object_map:
            locs = object_map[key]['locs']
            x_vals, y_vals = list(zip(*locs))
            object_map[key]['x_min'] = np.min(x_vals)
            object_map[key]['x_max'] = np.max(x_vals)
            object_map[key]['x_mean'] = (object_map[key]['x_min'] +  \
                    object_map[key]['x_max'])/2.0
            object_map[key]['y_min'] = np.min(y_vals)
            object_map[key]['y_max'] = np.max(y_vals)
            object_map[key]['y_mean'] = (object_map[key]['y_min'] +  \
                    object_map[key]['y_max'])/2.0

        return object_map

    def _update_cmap_attrs(self):
        '''
        Update the cmap for the agents so that there are more distractor object.
        '''

        attr_dims = (3,8,2,2)

        def is_unique_in_env(new_attrs):

            for old_attrs in [val['attr'] for key, val in self._obj_map.items()]:

                if np.all(old_attrs == new_attrs):
                    return False

            return True

        def attr_to_int(attr):
            """
            attr is a list of attribute values for an object
            """

            base = 1
            v = 0
            for dim, dim_size in reversed(list(enumerate([3,8,2,2]))) :
                v += attr[dim] * base
                base = base * dim_size
            return v

        for a in range(self.num_agents):
            atar = self.targets[0][a]
            non_goals = [o for o in self._obj_list if o != atar]

            # Sample number of distractor objects

            non_goals = np.random.permutation(non_goals)

            self.max_distractors = min(self.max_distractors, len(non_goals))

            for oidx in non_goals[:self.max_distractors]:
                akeep = np.random.choice(4, 3, replace=False)
                goal_attrs = self._obj_map[atar]['attr']

                # This is the new attribute for the object
                obj_attrs = np.zeros(4, dtype=int)
                obj_attrs[akeep] = goal_attrs[akeep]

                achange = list(set(np.arange(4, dtype=int)) - set(akeep))[0]

                sample_count = 0
                obj_done = False

                while(sample_count < self._SAMPLE_LIMIT):
                    sample_count += 1

                    # Sample new attribute value s
                    achange_val = np.random.choice(attr_dims[achange])

                    new_attr = obj_attrs.copy()
                    new_attr[achange] = achange_val

                    if is_unique_in_env(new_attr):

                        # Sample a non_goal object to assign this attribute to
                        # [NOTE]: Can be easily extended multiple objects being
                        # new values
                        o_val = self._obj_map[oidx]
                        old_aidx = o_val['attr']
                        new_aidx = new_attr
                        # Set previous values ot 0 at this object location
                        for l in o_val['locs']:
                            self.per_agent_cell_attr_map[a][l + tuple(old_aidx)] = 0

                        for l in o_val['locs']:
                            self.per_agent_cell_attr_map[a][l + tuple(new_aidx)] = 1

                        self._obj_map[oidx]['attr'] = new_attr

                        break

    def _get_full_obs(self, type='full'):
        """
        Get environment attribute grid
        """

        if type == 'full':
            H, W = self.cell_attr_map.shape[:2]
            cmap = self.per_agent_cell_attr_map.reshape(self.num_agents, H, W, -1)
        else:
            raise NotImplementedError

        return cmap
