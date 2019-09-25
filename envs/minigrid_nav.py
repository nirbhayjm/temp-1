from typing import Dict, Optional, Tuple, List, NewType

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from collections import namedtuple

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

XYTuple = namedtuple("XYTuple", ["x", "y"])

# MINIMAL_OBJECT_TO_IDX = {
#     'empty'         : 0,
#     'wall'          : 1,
#     'ball'          : 2,
#     'agent'         : 3,
# }
#
# MINIMAL_IDX_TO_OBJECT = dict(
#     zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
#
# MINIMAL_ENCODE = lambda idx: MINIMAL_OBJECT_TO_IDX[IDX_TO_OBJECT[idx]]
# MINIMAL_DECODE = lambda idx: OBJECT_TO_IDX[MINIMAL_IDX_TO_OBJECT[idx]]

Observation = NewType('Observation', Dict[str, np.ndarray])

class NavObject(Ball):
    '''An navigable cell (TODO: Add attributes)'''
    def __init__(self,
                 color: str = 'blue'):
        super().__init__(color=color)

    def can_overlap(self) -> bool:
        return True

    def can_pickup(self) -> bool:
        return False


class NavGridEnv(MiniGridEnv):
    """
    Point navigation wrapper for MiniGridEnv with fully observable
    state space and cardinal actions
    """

    class CardinalActions(IntEnum):
        # Cardinal movement
        right = 0
        down = 1
        left = 2
        up = 3

        # Done completing task
        done = 4

    def __init__(
        self,
        reward_type: str = 'reach_and_stop',
        grid_size: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        use_grid_in_state: bool = False,
        normalize_agent_pos: bool = False,
        reward_scale: float = 1.0,
        obs_alpha: float = 0.001,
        reset_prob: float = 1.0,
        term_prob: float = 0.0,
        render_rgb: bool = False,
        perturb_prob: float = 0.2,
        static_grid: bool = True,
        config_seed: int = 123,
        seed: int = 1337,
    ):
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Override MiniGridEnv actions
        self.actions = NavGridEnv.CardinalActions
        self.move_actions = [
            self.actions.right,
            self.actions.down,
            self.actions.left,
            self.actions.up,
        ]

        self.action_space = spaces.Discrete(len(self.actions))

        # self.encoding_range = len(MINIMAL_OBJECT_TO_IDX.keys())
        self.encoding_range = len(OBJECT_TO_IDX.keys())

        self.agent_pos_observation = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype='float',
        )

        if use_grid_in_state:
            self.grid_observation = spaces.Box(
                low=0,
                high=1,
                shape=(width, height, self.encoding_range),
                dtype='float',
            )
            # self.observation_space['image'] = self.grid_observation
            self.observation_space = spaces.Dict({
                'image': self.grid_observation,
                'agent_pos': self.agent_pos_observation,
            })

        else:
            self.observation_space = spaces.Dict({
                'agent_pos': self.agent_pos_observation,
            })

        self._obs_alpha = obs_alpha
        self._obs_mean = {key:np.zeros(self.observation_space.spaces[key].shape)\
            for key in self.observation_space.spaces}
        self._obs_var = {key:np.ones(self.observation_space.spaces[key].shape)\
            for key in self.observation_space.spaces}

        # Range of possible rewards
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.reward_range = (0, 1 * self.reward_scale)
        if self.reward_type == 'reach_and_stop':
            self.reward = self.reward_reach_and_stop

        # Random agent perturbation
        assert perturb_prob >= 0.0 and perturb_prob < 1.0
        self.perturb_prob = perturb_prob

        # Random episode termination
        assert term_prob >= 0.0 and term_prob < 1.0
        self.term_prob = term_prob

        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None

        # Renderer used to render observations (small-scale agent view)
        self.obs_render = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reset_prob = reset_prob
        self.first_reset = False

        self.see_through_walls = see_through_walls
        self.use_grid_in_state = use_grid_in_state
        self.normalize_agent_pos = normalize_agent_pos
        self.static_grid = static_grid
        self.grid_generated = False

        # Starting position and direction for the agent
        self.start_pos = None
        self.start_dir = None

        self._done = False

        # Initialize the RNG
        self.seed(seed=seed)
        self.config_seed = config_seed
        self.config_rng = np.random.RandomState()
        self.config_rng.seed(self.config_seed)

        # Rendering
        self.render_rgb = render_rgb
        self.CELL_PIXELS = CELL_PIXELS
        self.render_shape = (self.width * self.CELL_PIXELS,
                             self.height * self.CELL_PIXELS, 3)

        # Initialize the state
        self.reset()

    def seed(self, seed):
        self._seed = seed
        return super().seed(seed)

    def seed_config(self, config_seed):
        self.config_seed = config_seed
        self.config_rng = np.random.RandomState()
        self.config_rng.seed(self.config_seed)

    def reset_config_rng(self):
        self.config_rng = np.random.RandomState()
        self.config_rng.seed(self.config_seed)

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, value):
        # print("Setting value: {}".format(value))
        self._done = value

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

    def around_pos(self, dir: int) -> Tuple[int]:
        """
        Get the absolutie position of one of the 4 cardinal
        cells around agent as specified by dir
        """
        assert dir >= 0 and dir < 4
        pos = self.agent_pos + DIR_TO_VEC[dir]
        pos[0] = pos[0].clip(0, self.width - 1)
        pos[1] = pos[1].clip(0, self.height - 1)
        return pos

    def reset(self):
        do_reset = self._rand_float(0.0, 1.0) < self.reset_prob

        if do_reset or not self.first_reset:
            self.first_reset = True

            if not self.static_grid or not self.grid_generated:
                # Generate a new random grid at the start of each episode
                # To keep the same grid for each episode, call env.seed() with
                # the same seed before calling env.reset()
                self._gen_grid(self.width, self.height)
                self.grid_generated = True

            if hasattr(self, 'reset_agent_pos'):
                self.reset_agent_pos()
            else:
                # These fields should be defined by _gen_grid
                assert self.start_pos is not None
                # assert self.start_dir is not None

                # Check that the agent doesn't overlap with an object
                start_cell = self.grid.get(*self.start_pos)
                assert start_cell is None or start_cell.can_overlap()

                # Place the agent in the starting position and direction
                self.agent_pos = self.start_pos

                self.start_dir = 0
                self.agent_dir = self.start_dir

            # Item picked up, being carried, initially nothing
            self.carrying = None

        self.visit_count_grid = np.ones(
            (self.width, self.height), dtype='float')
        visit_count = self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]]

        # Step count since episode start
        self.step_count = 0

        self.done = False
        self.success = False

        if self.reward_type == 'reach_and_stop':
            self.goal_hit = False
        else:
            raise ValueError

        info = {
            'done': self.done,
            'visit_count': visit_count,
            'agent_pos': np.array(self.agent_pos),
            'agent_dir': np.array(self.agent_dir),
        }

        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')

        # Return first observation
        obs = self.gen_obs()
        return obs, info

    def encode_grid(self):
        orig_image = self.grid.encode(vis_mask=None)
        final_image = orig_image
        # final_image = np.copy(orig_image)

        # for i in range(self.grid.width):
        #     for j in range(self.grid.height):
        #         final_image[i, j, 0] = \
        #             MINIMAL_ENCODE(orig_image[i, j, 0])
        #
        # final_image[self.agent_pos[0], self.agent_pos[1], 0] = \
        #     MINIMAL_OBJECT_TO_IDX['agent']
        final_image = final_image[:, :, 0]

        one_hot_img = np.eye(self.encoding_range)[final_image]
        return one_hot_img.astype(np.float32)

    def is_goal(self, pos: np.ndarray) -> int:
        '''pos == mission'''
        assert type(self.mission) == np.ndarray
        return (pos == self.mission).min()

    def is_not_goal(self, pos: np.ndarray) -> bool:
        '''pos != mission and a NavObject is at pos'''
        assert type(self.mission) == np.ndarray
        if not self.is_goal(pos) and isinstance(
            self.grid.get(pos[0], pos[1]), NavObject):
            return True
        return False

    def reward_reach_and_stop(self, action: np.ndarray) -> float:
        target_reached_bonus = \
            1 - 0.9 * (self.step_count / self.max_steps)

        target_reached_bonus *= self.reward_scale

        if action == self.actions.done:
            return target_reached_bonus * 0.1

        elif not self.goal_hit:
            self.goal_hit = True
            return target_reached_bonus * 0.9

        else:
            return 0.0

    def normalize_pos(self, pos: Tuple[int]) -> np.ndarray:
        _x = (1.0 * pos[0]) / self.width
        _y = (1.0 * pos[1]) / self.height
        return np.array([_x, _y])

    def normalize_float_obs(self, obs: Observation, keys: List[str]):
        normalized_obs = {}
        for key in obs.keys():
            if key not in keys:
                normalized_obs[key] = obs[key]
                continue
            assert key in self.observation_space.spaces
            assert self.observation_space.spaces[key].dtype == 'float'
            self._obs_mean[key] = (1 - self._obs_alpha) * self._obs_mean[key] + \
                self._obs_alpha * obs[key]
            self._obs_var[key] = (1 - self._obs_alpha) * self._obs_var[key] + \
                self._obs_alpha * np.square(obs[key] - self._obs_mean[key])

            normalized_obs[key] = (obs[key] - \
                self._obs_mean[key]) / (np.sqrt(self._obs_var[key]) + 1e-8)
        return normalized_obs

    def gen_obs(self) -> Observation:
        """
        Generate the agent's view (fully observable grid)
        """

        # grid, vis_mask = self.gen_obs_grid()

        # image = self.grid.encode(vis_mask=None)
        if self.normalize_agent_pos:
            agent_pos = self.normalize_pos(self.agent_pos)
        else:
            agent_pos = np.array(self.agent_pos)
        #mission_pos = self.normalize_pos(self.mission)

        assert hasattr(self, 'mission'), \
            "environments must define a mission"

        # Observations are dictionaries containing:
        # - an image (fully observable view of the environment)
        # - a textual mission string (instructions for the agent)
        obs = {
            # 'image': image,
            # 'direction': self.agent_dir,
            'agent_pos': agent_pos.astype(np.float32),
            'mission': self.mission,
        }

        if self.use_grid_in_state:
            # Encode the fully observable view into a numpy array
            image = self.encode_grid()
            obs['image'] = image

        obs = self.normalize_float_obs(obs, keys=['agent_pos', 'image'])
        for key in obs:
            if obs[key].dtype == 'float':
                # Use float32 instead of float64
                obs[key] = obs[key].astype('float32')
        return obs

    def step(self, action: np.ndarray) \
        -> Tuple[Observation, np.ndarray, bool, Dict]:
        # action += 1
        self.step_count += 1

        reward = 0

        if not self.done:
            # Cardinal movement
            if action in self.move_actions:
                move_pos = self.around_pos(action)
                fwd_cell = self.grid.get(*move_pos)
                self.agent_dir = (action - 1) % 4

                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agent_pos = move_pos

                if fwd_cell != None and self.is_goal(move_pos):
                    reward += self.reward(action)

                elif fwd_cell != None and self.is_not_goal(move_pos):
                    # reward += self.reward(action) * 0.05
                    pass
                    penalty = self.reward(action) * 0.5
                    reward += -1 * penalty

                if fwd_cell != None and fwd_cell.type == 'lava':
                    self.done = True

            # Done action (not used by default)
            elif action == self.actions.done:
                if self.term_prob <= 0.0:
                    self.done = True
                # if self.success == False:
                #     curr_cell = self.agent_pos
                #     if self.is_goal(curr_cell):
                #         reward += self.reward(action)
                #         # print("Adding done reward!")
                #         self.success = True
                #
                #     elif self.is_not_goal(curr_cell):
                #         # reward += self.reward(action) * 0.05
                #         penalty = self.reward(action) * 0.5
                #         reward += -1 * penalty
                #         pass

            else:
                raise ValueError("{}".format(action))

        # Random episode termination
        if self._rand_float(0.0, 1.0) < self.term_prob:
            # print("Random term!")
            self.done = True

        # Stochastic environment transitions
        if self._rand_float(0.0, 1.0) < self.perturb_prob:
            # print("Random perturbation!")
            self.perturb_agent_pos()

        # One-time success-based reward
        if self.success == False:
            if self.is_goal(self.agent_pos):
                reward += self.reward(action)
                # print("Adding done reward!")
                self.success = True

        if self.step_count >= self.max_steps - 1:
            # print("Max Steps Exceeded.")
            self.done = True

        self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]] += 1
        visit_count = self.visit_count_grid[
            self.agent_pos[0], self.agent_pos[1]]

        obs = self.gen_obs()

        info = {
            'done': self.done,
            'visit_count': visit_count,
            'agent_pos': np.array(self.agent_pos),
            'agent_dir': np.array(self.agent_dir),
        }

        if self.render_rgb:
            info['rgb_grid'] = self.render(mode='rgb_array')

        if self.done:
            info.update({
                'image': self.encode_grid(),
                'success': self.success,
                'agent_pos': self.agent_pos,
            })

        return obs, reward, False, info

    def perturb_agent_pos(self):
        # Cardinal movement
        perturb_action = self.np_random.choice(self.move_actions)
        move_pos = self.around_pos(perturb_action)
        fwd_cell = self.grid.get(*move_pos)
        # self.agent_dir = perturb_action - 1
        if fwd_cell == None or fwd_cell.can_overlap():
            self.agent_pos = move_pos

        if fwd_cell != None and fwd_cell.type == 'lava':
            self.done = True

    def render(self, mode='human', close=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None:
            from gym_minigrid.rendering import Renderer
            self.grid_render = Renderer(
                self.width * CELL_PIXELS,
                self.height * CELL_PIXELS,
                True if mode == 'human' else False
            )

        r = self.grid_render

        if r.window:
            r.window.setText(self.mission)

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, CELL_PIXELS)

        # Draw the agent
        r.push()
        r.translate(
            CELL_PIXELS * (self.agent_pos[0] + 0.5),
            CELL_PIXELS * (self.agent_pos[1] + 0.5)
        )
        r.rotate(self.agent_dir * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 0),
            (0, 12),
            (12,  0),
            (0, -12),
        ])
        r.pop()
        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()

        return r

    def enumerate_states(self):
        old_agent_pos = self.agent_pos
        all_obs = {}
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.grid.get(i, j) != None:
                    continue
                _key = XYTuple(x = i, y = j)
                self.agent_pos = np.array([i, j])
                all_obs[_key] = self.gen_obs()
        self.agent_pos = old_agent_pos
        return all_obs

    def place_agent(self, rng=None):
        if not hasattr(self, '_occupancy_grid'):
            # Occupancy grid, has 1s where agent can't be placed
            self._occupancy_grid = np.ones((self.width, self.height))
            for row in range(self.width):
                for col in range(self.height):
                    cell = self.grid.get(row, col)
                    # assert start_cell is None or start_cell.can_overlap()
                    if cell is None or cell.can_overlap():
                        self._occupancy_grid[row, col] = 0
            self._unoccupied_x, self._unoccupied_y = \
                np.where(self._occupancy_grid == 0)

            assert len(self._unoccupied_x) > 0

        if rng == None:
            rng = self.np_random

        start_index = rng.randint(len(self._unoccupied_x))
        self.start_pos = (
            self._unoccupied_x[start_index],
            self._unoccupied_y[start_index],
        )
        self.start_dir = 0
