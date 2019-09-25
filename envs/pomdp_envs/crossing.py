from typing import Dict, Optional, Tuple, List, NewType

from gym_minigrid.minigrid import *
# from gym_minigrid.envs import CrossingEnv
from ..mdp_envs.maze import MazeBase

# from ..minigrid_nav import *

CELL_PIXELS = 8

STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}

class CrossingEnvWrapper(MiniGridEnv):
    """
    Simple grid world for point navigation
    """

    def __init__(
        self,
        size=9,
        # num_rows=2,
        max_steps=10,
        obstacle_type='wall',
        grid_type='crossing',
        num_crossings=3,
        obs_win_size=1,
        reset_on_done=False,
        # use_grid_in_state=False,
        # normalize_agent_pos=False,
        obs_alpha=0.001,
        reward_scale=1.0,
        static_env_grid=1,
        spawn='center',
        corridor_len=3,
        render_rgb = False,
        transfer_mode=False,
        term_prob=0.0,
        reset_prob=1.0,
        perturb_prob=0.0,
        agent_view_size=7,
        randomize_goal_pos=True,
        end_on_goal=False,
        complexity=0.1,
        density=0.1,
        config_seed=1234,
        seed=123,
    ):
        self.obs_win_size = obs_win_size
        self.config_seed = config_seed
        self.spawn = spawn
        self.render_rgb = render_rgb
        self.corridor_len = corridor_len
        self.grid_type = grid_type
        self.complexity = complexity
        self.density = density
        self.reward_scale = reward_scale
        self.end_on_goal = end_on_goal
        self.transfer_mode = transfer_mode
        self.randomize_goal_pos = randomize_goal_pos
        self.reset_on_done = reset_on_done

        self.num_channels = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)
        # self.num_channels = 3

        # Termination probability at every time step
        assert term_prob < 1.0 and term_prob >= 0.0
        self.term_prob = term_prob

        # Environment reset probability at every episode term
        assert reset_prob >= 0.0 and reset_prob <= 1.0
        self.reset_prob = reset_prob
        self.first_reset = False

        # Random agent perturbation
        assert perturb_prob >= 0.0 and perturb_prob < 1.0
        self.perturb_prob = perturb_prob

        self.static_grid = static_env_grid
        self.grid_generated = False

        assert obstacle_type in ['wall', 'lava']
        if obstacle_type == 'wall':
            obstacle_type = Wall
        else:
            obstacle_type = Lava

        # Config seed
        self.seed_config(config_seed)

        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        MiniGridEnv.__init__(
            self,
            # size=size,
            # num_crossings=num_crossings,
            # obstacle_type=obstacle_type,
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed,
        )

        # Seeding
        self.seed(seed=self.config_seed)
        self.config_rng.seed(self.config_seed)
        self.grid_generated = False
        self.first_reset = False
        self.reset()
        self.seed(seed=seed)

        new_spaces = self.observation_space.spaces
        new_spaces.pop('image')
        new_spaces.update({
            'image': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.agent_view_size, self.agent_view_size,
                    self.num_channels * self.obs_win_size),
                dtype='float',
            ),
            'direction': spaces.Box(
                low=0,
                high=3,
                shape=(4 * self.obs_win_size,),
                dtype='float',
            ),
            'pos': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2 * self.obs_win_size,),
                dtype='float',
            ),
            'mission': spaces.Box(
                low=0,
                high=size,
                shape=(2 * self.obs_win_size,),
                dtype='float',
            ),
            'goal_vector': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2 * self.obs_win_size,),
                dtype='float',
            ),
        })
        self.observation_space = spaces.Dict(new_spaces)
        self.init_obs_queue()

        # self._obs_alpha = obs_alpha
        # self._obs_mean = {key:np.zeros(self.observation_space.spaces[key].shape)\
        #     for key in self.observation_space.spaces}
        # self._obs_var = {key:np.ones(self.observation_space.spaces[key].shape)\
        #     for key in self.observation_space.spaces}

        self.max_steps = max_steps

        self.done = False

        self.CELL_PIXELS = CELL_PIXELS
        self.render_shape = (self.width * self.CELL_PIXELS,
                             self.height * self.CELL_PIXELS, 3)

    def horz_wall(self, x, y, length=None, lava_or_wall='wall'):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            if lava_or_wall == 'lava':
                self.grid.set(x + i, y, Lava())
            elif lava_or_wall == 'wall':
                self.grid.set(x + i, y, Wall())
            else:
                raise ValueError

    def vert_wall(self, x, y, length=None, lava_or_wall='wall'):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            if lava_or_wall == 'lava':
                self.grid.set(x, y + j, Lava())
            elif lava_or_wall == 'wall':
                self.grid.set(x, y + j, Wall())
            else:
                raise ValueError

    def seed_config(self, config_seed):
        self.config_seed = config_seed
        self.config_rng = np.random.RandomState()
        self.config_rng.seed(self.config_seed)

    def reset_config_rng(self):
        self.config_rng = np.random.RandomState()
        self.config_rng.seed(self.config_seed)

    def init_obs_queue(self):
        self._obs_queue = {}
        for key in self.observation_space.spaces:
            dtype = self.observation_space.spaces[key].dtype
            if dtype == 'float64':
                dtype = 'float32'

            self._obs_queue[key] = np.zeros(
                self.observation_space.spaces[key].shape,
                dtype=dtype)

    @property
    def reset_prob(self):
        return self._reset_prob

    @reset_prob.setter
    def reset_prob(self, value):
        assert value <= 1.0 and value >= 0.0
        self._reset_prob = value

    def wall_rect_filled(self, x, y, w, h):
        for i in range(0, w):
            for j in range(0, h):
                self.grid.set(x + i, y + j, Wall())

    def modify_attr(self, attr, value):
        assert hasattr(self, attr)
        setattr(self, attr, value)

    def _gen_grid_empty(self, width, height):
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(width - 2, height - 2, Goal())
        # self.grid.set(3, 3, Goal())

    def _gen_grid_tworooms(self, width, height):
        # Draw rectangular wall around the grid
        self.grid.wall_rect(0, 0, width, height)

        rw = width - 2 - self.corridor_len

        # rw is the combined width of left and right room
        assert rw >= 2, "Corridor Length too big for the Environment."

        corr_x = rw // 2 + 1
        corr_y = (height - 1) // 2

        self.wall_rect_filled(
            corr_x, 0, self.corridor_len , corr_y)
        self.wall_rect_filled(
            corr_x, corr_y + 1, self.corridor_len, height - corr_y - 1)

        self.center_pos = self.width // 2, corr_y
        self.grid.set(width - 2, height - 2, Goal())

    def _gen_grid_fourrooms(self, width, height):
        # Draw rectangular wall around the grid
        self.grid.wall_rect(0, 0, width, height)


        cx = 1 + ((width - 2) // 2)
        cy = 1 + ((height - 2) // 2)

        self.grid.horz_wall(0, cy)
        self.grid.vert_wall(cx, 0)

        # Get position of the doors

        doors = [(cx // 2, cy),
                (cx + cx // 2, cy),
                (cx, cy // 2),
                (cx, cy + cy // 2)]

        [self.grid.set(*pos, None) for pos in doors]

        self.grid.set(width - 2, height - 2, Goal())
        # if not hasattr(self, '_occupancy_grid'):
        #     self.gen_occupancy_grid()
        #
        # g_index = self.config_rng.randint(len(self._unoccupied_x))
        # self.grid.set(
        #     self._unoccupied_x[g_index],
        #     self._unoccupied_y[g_index],
        #     Goal(),
        # )
        pass

    def _gen_grid_maze(self, width, height):
        maze = MazeBase.create_maze(
            height,
            width,
            rng=self.config_rng,
            complexity=self.complexity,
            density=self.density,
        )

        height, width = maze.board.shape
        # Draw rectangular wall around the grid
        for i in range(height):
            for j in range(width):
                if(maze.board[i][j] == 0):
                    self.grid.set(j, height - 1 - i, Wall())

        if not hasattr(self, '_occupancy_grid'):
            self.gen_occupancy_grid()

        g_index = self.config_rng.randint(len(self._unoccupied_x))
        goal_x = self._unoccupied_x[g_index]
        goal_y = self._unoccupied_y[g_index]
        self.grid.set(goal_x, goal_y, Goal())
        self.goal_pos = np.array((goal_x, goal_y))

    def _gen_grid_half_danger(self, width, height):
        # Draw rectangular wall around the grid
        self.grid.wall_rect(0, 0, width, height)

        # self.horz_wall(2, 1,  width - 4, lava_or_wall='lava')

        half_height = 1 + ((height - 2) // 2)
        half_width = 1 + ((width - 2) // 2)

        for c_idx in range(half_height + 1, height - 1, 2):
            for r_idx in range(2, width - 1, 2):
                self.grid.set(r_idx, c_idx, Lava())

        if self.transfer_mode:
            self.fixed_goal_pos = (half_width, half_height)
        else:
            self.fixed_goal_pos = (width - 2, 1)
        # self.grid.set(*self.goal_pos, Goal())

    def _gen_grid(self, width, height):
        if not self.static_grid or not self.grid_generated\
        or self.grid_type == 'crossing':
            # super()._gen_grid(width, height)
            self.grid = Grid(width, height)

            if self.grid_type == 'empty':
                self._gen_grid_empty(width, height)
            elif self.grid_type == 'two-room-corridor':
                self._gen_grid_tworooms(width, height)
            elif self.grid_type == 'four-rooms':
                self._gen_grid_fourrooms(width, height)
            elif self.grid_type == 'maze':
                self._gen_grid_maze(width, height)
            elif self.grid_type == 'crossing':
                self._gen_grid_half_danger(width, height)
            else:
                raise ValueError
            self.grid_generated = True

        if not hasattr(self, '_occupancy_grid'):
            self.gen_occupancy_grid()

        if self.grid_type == 'crossing':
            if self.transfer_mode:
                self.start_pos = (1, 1)
                self.start_dir = self._rand_int(0, 4)
                self.goal_pos = np.array(self.fixed_goal_pos)
            else:
                self.start_pos = (width // 2, 1)
                self.start_dir = self._rand_int(0, 4)

                if self.randomize_goal_pos:
                    g_index = self.config_rng.randint(
                        len(self._unoccupied_x))
                    goal_x = self._unoccupied_x[g_index]
                    goal_y = self._unoccupied_y[g_index]
                    self.goal_pos = np.array((goal_x, goal_y))
                else:
                    self.goal_pos = np.array(self.fixed_goal_pos)

            self.grid.set(*self.goal_pos, Goal())

        elif self.spawn == 'center':
            if self.grid_type == 'two-room-corridor':
                self.start_pos = self.center_pos
            else:
                self.start_pos = (1, 1)
            self.start_dir = 0
        else:
            if self.spawn == 'fixed':
                rng = self.config_rng
                self.start_dir = self._rand_int(0, 4)
            else:
                rng = None
                self.start_dir = 0
            # self.start_pos = (
            #     self._rand_int(0, width),
            #     self._rand_int(0, height)
            # )
            self.place_agent(rng=rng)

        if not hasattr(self, '_occupancy_grid'):
            self.gen_occupancy_grid()

        self.mission = np.zeros(2)

    def reset(self):
        do_reset = self._rand_float(0.0, 1.0) <= self.reset_prob

        if do_reset or not self.first_reset:
            self.first_reset = True
            obs = super().reset()
        else:
            obs = self.gen_obs()

        self.step_count = 0
        self.episode_reward = 0

        self.visit_count_grid = np.ones(
            (self.width, self.height), dtype='float')
        visit_count = self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]]
        self.done = False
        info = {
            'agent_pos': np.array(self.agent_pos),
            'agent_dir': np.array(self.agent_dir),
            'visit_count': visit_count,
            'current_room': 0,
            'max_room_id': 0,
            'step_count': self.step_count,
            'is_heuristic_ds': False,
        }
        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')

        return obs, info

    def reset_agent_pos(self):
        if self.spawn == 'random':
            self.place_agent()
            self.start_dir = self._rand_int(0, 4)
            # self.start_dir = 0

        else:
            # if self.spawn == 'center':
            # Place the agent in the top-left corner
            self.start_pos = (1, 1)
            self.start_dir = 0

        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir

    def around_pos(self, dir: int) -> Tuple[int]:
        """
        Get the absolute position of one of the 4 cardinal
        cells around agent as specified by dir
        """
        assert dir >= 0 and dir < 4
        pos = self.agent_pos + DIR_TO_VEC[dir]
        pos[0] = pos[0].clip(0, self.width - 1)
        pos[1] = pos[1].clip(0, self.height - 1)
        return pos

    def perturb_agent_pos(self):
        # Cardinal movement
        perturb_dir = self.np_random.choice(4)
        perturb_orientation = self.np_random.choice(3) - 1
        self.agent_dir = (self.agent_dir + perturb_orientation) % 4
        move_pos = self.around_pos(perturb_dir)
        fwd_cell = self.grid.get(*move_pos)
        # self.agent_dir = perturb_dir - 1
        if fwd_cell == None or fwd_cell.can_overlap():
            self.agent_pos = move_pos

        if fwd_cell != None and fwd_cell.type == 'lava':
            self.done = True

    def encode_one_hot_image(self, image):
        img_obj = image[:, :, 0]
        img_colors = image[:, :, 1]
        img_state = image[:, :, 2]

        img_obj = np.eye(len(OBJECT_TO_IDX))[img_obj]
        img_colors = np.eye(len(COLOR_TO_IDX))[img_colors]
        img_state = np.eye(len(STATE_TO_IDX))[img_state]

        img_cat = np.concatenate([img_obj, img_colors, img_state], 2)
        return img_cat

    def gen_obs(self):
        obs = super().gen_obs()
        if self.num_channels != 3:
            obs['image'] = self.encode_one_hot_image(obs['image']).astype('float')
        obs['direction'] = np.eye(4)[obs['direction']]
        # obs['pos'] = 1.0 * np.array(self.agent_pos) - np.array(self.start_pos)
        obs['pos'] = 1.0 * np.array(self.agent_pos)
        obs['goal_vector'] = (self.goal_pos - self.agent_pos) / 25
        # obs['pos'] /= np.sqrt(self.grid.height * self.grid.width)
        obs = self.normalize_float_obs(obs, selected_keys=['image'])
        for key in obs.keys():
            if hasattr(obs[key], 'dtype') and obs[key].dtype == 'float':
                # Use float32 instead of float64
                obs[key] = obs[key].astype('float32')

        NC = obs['image'].shape[2]
        # WS = self.obs_win_size

        if not hasattr(self, '_obs_queue'):
            return None

        # Shift obs by 1 * image_channels
        self._obs_queue['image'][:, :, NC:] = \
            self._obs_queue['image'][:, :, :-NC]
        # Update obs at head of queue
        self._obs_queue['image'][:, :, :NC] = obs['image']

        remaining_keys = [key for key in obs.keys() if key != 'image']

        for key in remaining_keys:
            assert len(obs[key].shape) == 1
            SH = obs[key].shape[0]
            # Shift obs by 1 * SH
            self._obs_queue[key][SH:] = \
                self._obs_queue[key][:-SH]
            # Update obs at head of queue
            self._obs_queue[key][:SH] = obs[key]

        # print("Direction in obs_queue:")
        # print(self._obs_queue['direction'])
        return self._obs_queue

    def normalize_float_obs(self, obs, selected_keys):
        # # Do nothing if _obs_alpha undefined
        # if not hasattr(self, '_obs_alpha'):
        #     return obs
        #
        # normalized_obs = {}
        # for key in obs.keys():
        #     # Skip non-selected keys
        #     if key not in selected_keys:
        #         normalized_obs[key] = obs[key]
        #         continue
        #
        #     # Validate input
        #     assert key in self.observation_space.spaces, "{}".format(key)
        #     assert obs[key].dtype == 'float',\
        #         "Expected '{}' to be of dtype '{}' buy got '{}'".format(
        #             key, 'float', obs[key].dtype)
        #
        #     # Normalize obs
        #     self._obs_mean[key] = (1 - self._obs_alpha) * self._obs_mean[key] + \
        #         self._obs_alpha * obs[key]
        #     self._obs_var[key] = (1 - self._obs_alpha) * self._obs_var[key] + \
        #         self._obs_alpha * np.square(obs[key] - self._obs_mean[key])
        #
        #     normalized_obs[key] = (obs[key] - \
        #         self._obs_mean[key]) / (np.sqrt(self._obs_var[key]) + 1e-8)
        # return normalized_obs

        return obs

    def gen_occupancy_grid(self):
        # Occupancy grid, has 1s where agent can't be placed
        self._occupancy_grid = np.ones((self.width, self.height))
        for row in range(self.width):
            for col in range(self.height):
                cell = self.grid.get(row, col)
                # assert start_cell is None or start_cell.can_overlap()
                # if cell is None or cell.can_overlap():
                if cell is None:
                    self._occupancy_grid[row, col] = 0
        self._unoccupied_x, self._unoccupied_y = \
            np.where(self._occupancy_grid == 0)
        assert len(self._unoccupied_x) > 0

    def place_agent(self, rng=None):
        if not hasattr(self, '_occupancy_grid') or not self.static_grid:
            self.gen_occupancy_grid()

        if rng == None:
            rng = self.np_random

        start_index = rng.randint(len(self._unoccupied_x))
        self.start_pos = (
            self._unoccupied_x[start_index],
            self._unoccupied_y[start_index],
        )
        self.start_dir = 0

    def step(self, action):
        done = False

        obs, reward, env_done, _ = super().step(action)

        if self.end_on_goal and env_done:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        reward = self.reward_scale * reward

        agent_cell = self.grid.get(*self.agent_pos)
        if agent_cell != None and agent_cell.type == 'lava':
            done = True
            reward *= 0

        # Stochastic environment transitions
        if self._rand_float(0.0, 1.0) < self.perturb_prob:
            # print("Random perturbation!")
            if self.grid_type == 'crossing':
                if action == self.actions.forward:
                    self.perturb_agent_pos()
                else:
                    pass
            else:
                self.perturb_agent_pos()

        self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]] += 1
        visit_count = self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]]

        if self._rand_float(0.0, 1.0) < self.term_prob:
            # print("Random term!")
            # Random episode termination
            done = True

        info = {
            'agent_pos': np.array(self.agent_pos),
            'agent_dir': np.array(self.agent_dir),
            'current_room': 0,
            'is_heuristic_ds': False,
            'max_room_id': 0,
            'step_count': self.step_count,
            'visit_count': visit_count,
            'done': self.done,
        }

        self.episode_reward += reward
        if done:
            self.done = True
            info['episode_reward'] = self.episode_reward

        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')

        if self.reset_on_done:
            return obs, reward, done, info
        else:
            return obs, reward, False, info
