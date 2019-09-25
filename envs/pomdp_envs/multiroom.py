from gym_minigrid.minigrid import *
from gym_minigrid.envs import KeyCorridor
from gym_minigrid.envs import MultiRoomEnv

# from ..minigrid_nav import *

CELL_PIXELS = 8

STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}
class MultiroomWrapper(MultiRoomEnv):
    """
    Simple grid world for point navigation
    """

    def __init__(
        self,
        size=4,
        spawn='fixed',
        num_rooms=2,
        max_steps=10,
        obs_alpha=0.001,
        obs_win_size=1,
        reward_scale=1.0,
        render_rgb = False,
        term_prob=0.0,
        reset_prob=1.0,
        agent_view_size=3,
        static_env_grid=1,
        doors_open=False,
        end_on_goal=False,
        config_seed=1234,
        reset_on_done=False,
        use_heuristic_ds=False,
        seed=123,
    ):
        assert agent_view_size == 3
        self.obs_win_size = obs_win_size
        self.render_rgb = render_rgb
        self.reward_scale = reward_scale
        self.end_on_goal = end_on_goal
        self.spawn = spawn
        self.doors_open = doors_open
        self.reset_on_done = reset_on_done
        self.use_heuristic_ds = use_heuristic_ds
        self.heuristic_ds_list = [
            'corners',
            'doorways',
        ]

        # Termination probability at every time step
        assert term_prob < 1.0 and term_prob >= 0.0
        self.term_prob = term_prob

        # Environment reset probability at every episode term
        assert reset_prob >= 0.0 and reset_prob <= 1.0
        self.reset_prob = reset_prob
        self.first_reset = False

        self.static_grid = static_env_grid
        self.grid_generated = False

        self.num_channels = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        self.seed_config(config_seed)
        self.seed(seed=seed)

        super().__init__(
            minNumRooms=num_rooms,
            maxNumRooms=num_rooms,
            maxRoomSize=size,
        )

        self.agent_view_size = agent_view_size
        # Seeding
        self.seed(seed=seed)
        self.grid_generated = False
        self.first_reset = False
        self.reset()

        new_spaces = self.observation_space.spaces
        new_spaces.pop('image')
        new_spaces.update({
            'image': spaces.Box(
                low=0,
                high=255,
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
            'goal_vector': spaces.Box(
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
        })
        self.observation_space = spaces.Dict(new_spaces)
        self.init_obs_queue()

        # self._obs_alpha = obs_alpha
        # self._obs_mean = {key:np.zeros(self.observation_space.spaces[key].shape)\
        #     for key in self.observation_space.spaces}
        # self._obs_var = {key:np.ones(self.observation_space.spaces[key].shape)\
        #     for key in self.observation_space.spaces}

        # self.max_steps = 20 * num_rooms
        self.max_steps = max_steps

        self.done = False

        self.CELL_PIXELS = CELL_PIXELS
        self.render_shape = (self.width * self.CELL_PIXELS,
                             self.height * self.CELL_PIXELS, 3)

    def seed_config(self, config_seed):
        self.config_seed = config_seed
        self.config_rng = np.random.RandomState()
        self.config_rng.seed(self.config_seed)

    def reset_config_rng(self):
        self.config_rng = np.random.RandomState()
        self.config_rng.seed(self.config_seed)

    def seed(self, seed):
        return_value = super().seed(seed)
        self.default_rng = self.np_random
        return return_value

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

    def modify_attr(self, attr, value):
        assert hasattr(self, attr)
        setattr(self, attr, value)

    def _gen_grid(self, width, height):
        # if not self.static_grid or not self.grid_generated:
        #     super()._gen_grid(width, height)
        #     self.grid_generated = True

        if self.static_grid:
            self.config_rng.seed(self.config_seed)

        old_rng = self.np_random
        self.np_random = self.config_rng
        super()._gen_grid(width, height)
        self.np_random = old_rng

        if self.spawn == 'random':
            # random_room_id = self.default_rng.randint(0, len(self.rooms))
            # self.place_agent(
            #     self.rooms[random_room_id].top,
            #     self.rooms[random_room_id].size,
            # )
            self.place_agent(
                self.rooms[0].top,
                self.rooms[0].size,
            )

        if self.doors_open:
            for idx, room in enumerate(self.rooms):
                # print(f"Room {idx}; entry={room.entryDoorPos}; exit={room.exitDoorPos}")
                door_pos = room.entryDoorPos
                grid_cell = self.grid.get(*door_pos)
                if isinstance(grid_cell, Door) and not grid_cell.is_open:
                    # print("Open door found!")
                    grid_cell.is_open = True

        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir

        self.mission = np.zeros(2)

    def reset(self):
        do_reset = self._rand_float(0.0, 1.0) <= self.reset_prob

        if do_reset or not self.first_reset:
            self.first_reset = True
            old_rng = self.np_random
            self.np_random = self.config_rng
            obs = super().reset()
            self.np_random = old_rng
        else:
            obs = self.gen_obs()
            self.step_count = 0

        self.visit_count_grid = np.ones(
            (self.width, self.height), dtype='float')
        visit_count = self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]]

        self.done = False
        self.goal_reached = False

        self.max_room_id = 0
        self.current_room_id = 0
        self.episode_reward = 0

        info = {
            'agent_pos': np.array(self.agent_pos),
            'agent_dir': np.array(self.agent_dir),
            'current_room': self.agent_current_room(self.agent_pos),
            'visit_count': visit_count,
            'max_room_id': 0,
            'step_count': self.step_count,
            'is_heuristic_ds': False, # Start state is never heuristic ds
        }
        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')
            info['str_grid'] = np.array(self.__str__())

        return obs, info

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
            obs['image'] = self.encode_one_hot_image(
                obs['image']).astype('float')
        obs['direction'] = np.eye(4)[obs['direction']]
        # obs['pos'] = 1.0 * np.array(self.agent_pos) - np.array(self.start_pos)
        obs['pos'] = 1.0 * np.array(self.agent_pos)
        # obs['pos'] /= np.sqrt(self.grid.height * self.grid.width)
        obs['goal_vector'] = (self.goal_pos - self.agent_pos)/self.grid.width
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
        # Do nothing if _obs_alpha undefined
        #if not hasattr(self, '_obs_alpha'):
        #    return obs

        #normalized_obs = {}
        #for key in obs.keys():
        #    # Skip non-selected keys
        #    if key not in selected_keys:
        #        normalized_obs[key] = obs[key]
        #        continue

        #    # Validate input
        #    assert key in self.observation_space.spaces, "{}".format(key)
        #    assert obs[key].dtype == 'float',\
        #        "Expected '{}' to be of dtype '{}' buy got '{}'".format(
        #            key, 'float', obs[key].dtype)

        #    # Normalize obs
        #    self._obs_mean[key] = (1 - self._obs_alpha) * self._obs_mean[key] + \
        #        self._obs_alpha * obs[key]
        #    self._obs_var[key] = (1 - self._obs_alpha) * self._obs_var[key] + \
        #        self._obs_alpha * np.square(obs[key] - self._obs_mean[key])

        #    normalized_obs[key] = (obs[key] - \
        #        self._obs_mean[key]) / (np.sqrt(self._obs_var[key]) + 1e-8)
        #return normalized_obs

        return obs

    # def gen_occupancy_grid(self):
    #     # Occupancy grid, has 1s where agent can't be placed
    #     self._occupancy_grid = np.ones((self.width, self.height))
    #     for row in range(self.width):
    #         for col in range(self.height):
    #             cell = self.grid.get(row, col)
    #             # assert start_cell is None or start_cell.can_overlap()
    #             if cell is None or cell.can_overlap():
    #                 self._occupancy_grid[row, col] = 0
    #     self._unoccupied_x, self._unoccupied_y = \
    #         np.where(self._occupancy_grid == 0)
    #     assert len(self._unoccupied_x) > 0

    # def place_agent_randomly(self, rng=None):
    #     if not hasattr(self, '_occupancy_grid'):
    #         self.gen_occupancy_grid()
    #
    #     if rng == None:
    #         rng = self.np_random
    #
    #     start_index = rng.randint(len(self._unoccupied_x))
    #     self.start_pos = (
    #         self._unoccupied_x[start_index],
    #         self._unoccupied_y[start_index],
    #     )
    #     self.start_dir = 0

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf,
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.start_pos = None
        if self.spawn == 'random':
            old_rng = self.np_random
            self.np_random = self.default_rng

            pos = self.place_obj(None, top, size, max_tries=max_tries)

            self.start_pos = pos
            if rand_dir:
                self.start_dir = self._rand_int(0, 4)

            self.np_random = old_rng
        else:
            pos = self.place_obj(None, top, size, max_tries=max_tries)
            self.start_pos = pos
            # if rand_dir:
            #     self.start_dir = self._rand_int(0, 4)
            self.start_dir = 0

        return pos

    def agent_current_room(self, agent_pos):
        for room_id, room in enumerate(self.rooms):
            agent_x = agent_pos[0]
            agent_y = agent_pos[1]

            # if agent_x >
            room_low_x = room.top[0]
            room_low_y = room.top[1]

            room_high_x = room.top[0] + room.size[0] - 1
            room_high_y = room.top[1] + room.size[1] - 1

            if agent_x >= room_low_x \
            and agent_x <= room_high_x \
            and agent_y >= room_low_y \
            and agent_y <= room_high_y:
                self.max_room_id = max(room_id, self.max_room_id)
                self.current_room_id = room_id
                return room_id
        assert False

    def is_corner(self, current_room):
        agent_x = self.agent_pos[0]
        agent_y = self.agent_pos[1]

        room = self.rooms[current_room]
        room_low_x = room.top[0]
        room_low_y = room.top[1]

        room_high_x = room.top[0] + room.size[0] - 1
        room_high_y = room.top[1] + room.size[1] - 1

        is_corner = False

        # Bottom left
        if agent_x == room_low_x + 1 \
        and agent_y == room_low_y + 1:
            is_corner = True

        # Top left
        if agent_x == room_low_x + 1 \
        and agent_y == room_high_y - 1:
            is_corner = True

        # Bottom right
        if agent_x == room_high_x - 1 \
        and agent_y == room_low_y + 1:
            is_corner = True

        # Top right
        if agent_x == room_high_x - 1 \
        and agent_y == room_high_y - 1:
            is_corner = True

        return is_corner

    def is_doorway(self, current_room):
        agent_x = self.agent_pos[0]
        agent_y = self.agent_pos[1]

        room = self.rooms[current_room]
        room_low_x = room.top[0]
        room_low_y = room.top[1]

        room_high_x = room.top[0] + room.size[0] - 1
        room_high_y = room.top[1] + room.size[1] - 1

        is_doorway = False

        # Left column
        if agent_x == room_low_x:
            is_doorway = True

        # Right column
        if agent_x == room_high_x:
            is_doorway = True

        # Bottom row
        if agent_y == room_low_y:
            is_doorway = True

        # Top row
        if agent_y == room_high_y:
            is_doorway = True

        return is_doorway

    def is_heuristic_ds(self, current_room):
        is_heuristic_ds = False
        if self.use_heuristic_ds:
            is_corner = False
            if 'corners' in self.heuristic_ds_list:
                is_corner = self.is_corner(current_room)

            is_doorway = False
            if 'doorways' in self.heuristic_ds_list:
                is_doorway = self.is_doorway(current_room)

            is_heuristic_ds = is_corner or is_doorway
        return is_heuristic_ds

    def step(self, action):
        done = False
        obs, reward, env_done, _ = super().step(action)

        if self.end_on_goal and env_done:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        if self.goal_reached:
            reward *= 0

        if reward > 0:
            self.goal_reached = True

        reward = self.reward_scale * reward

        self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]] += 1
        visit_count = self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]]

        # # Stochastic environment transitions
        # if self._rand_float(0.0, 1.0) < self.perturb_prob:
        #     # print("Random perturbation!")
        #     self.perturb_agent_pos()

        if self._rand_float(0.0, 1.0) < self.term_prob:
            # print("Random term!")
            # Random episode termination
            done = True

        if done:
            self.done = True

        current_room = self.agent_current_room(self.agent_pos)
        info = {
            'agent_pos': np.array(self.agent_pos),
            'agent_dir': np.array(self.agent_dir),
            'current_room': current_room,
            'is_heuristic_ds': self.is_heuristic_ds(current_room),
            'max_room_id': self.max_room_id,
            'step_count': self.step_count,
            'visit_count': visit_count,
            'done': self.done,
        }

        self.episode_reward += reward
        if done:
            info['episode_reward'] = self.episode_reward

        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')
            info['str_grid'] = np.array(self.__str__())

        if self.reset_on_done:
            return obs, reward, done, info
        else:
            return obs, reward, False, info
