from gym_minigrid.minigrid import *
from ..minigrid_nav import *

import itertools as itt


class CrossingMDP(NavGridEnv):
    def __init__(
        self,
        size=5,
        num_crossings=1,
        obstacle_type='lava',
        max_steps=10,
        use_grid_in_state=False,
        normalize_agent_pos=False,
        obs_alpha=0.001,
        reward_scale=1.0,
        spawn='none',
        seed=123,
        reset_prob=1.0,
        perturb_prob=0.0,
        render_rgb=False,
        term_prob=0.0,
    ):
        assert obstacle_type in ['wall', 'lava']
        if obstacle_type == 'wall':
            self.obstacle_type = Wall
        else:
            self.obstacle_type = Lava
        self.num_crossings = num_crossings
        self.spawn = spawn

        super().__init__(
            grid_size=size,
            width=None,
            height=None,
            max_steps=max_steps,
            see_through_walls=False,
            use_grid_in_state=use_grid_in_state,
            normalize_agent_pos=normalize_agent_pos,
            obs_alpha=obs_alpha,
            reward_scale=reward_scale,
            reset_prob=reset_prob,
            perturb_prob=perturb_prob,
            term_prob=term_prob,
            render_rgb=render_rgb,
            seed=seed,
        )
        new_spaces = self.observation_space.spaces
        new_spaces.update({
            'mission': spaces.MultiDiscrete([size, size]),
        })
        self.observation_space = spaces.Dict(new_spaces)


    # def _gen_grid(self, width, height):
    #     # Create an empty grid
    #     self.grid = Grid(width, height)
    #
    #     # Generate the surrounding walls
    #     # NOTE: Do not generate surrounding walls
    #     # self.grid.wall_rect(0, 0, width, height)
    #
    #     half_height = (self.grid.height+1)//2
    #     center = (self.grid.height)//2
    #
    #     # Generate the wall in front of spawn location
    #     # self.grid.vert_wall(1, 1, length=half_height)
    #     # self.grid.horz_wall(1, center, length=width-1)
    #
    #     # Place the agent in the top-left corner
    #     if self.spawn == 'center':
    #         self.start_pos = (center, center)
    #     elif self.spawn == 'none':
    #         self.start_pos = (
    #             self._rand_int(0, width),
    #             self._rand_int(0, height)
    #         )
    #     self.start_dir = 0
    #
    #     corners = [
    #         [0, 0],
    #         [width - 1, 0],
    #         [0, height - 1],
    #         [width - 1, height - 1],
    #     ]
    #
    #     # object_pos = [
    #     #     [self._rand_int(0, width), self._rand_int(0, height)],
    #     #     [self._rand_int(0, width), self._rand_int(0, height)],
    #     # ]
    #
    #     _choice = np.random.choice(len(corners), 2, replace=False)
    #     # object_pos = [corners[self._rand_int(0,4)]]
    #     object_pos = [corners[idx] for idx in _choice]
    #
    #     # # Place two objects
    #     # self.grid.set(*object_pos[0], NavObject())
    #     # self.grid.set(*object_pos[1], NavObject())
    #
    #     # Flip a coin to decide which goal to select
    #     self.goal_index = 0
    #     # _rand = self._rand_int(0, 10)
    #     # self.goal_index = int(_rand < 3)
    #     # self.goal_index = int(self._rand_bool())
    #     self.mission = np.array(object_pos[self.goal_index])

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.grid.set(i, j, self.obstacle_type())

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.goal_index = 0
        self.mission = np.array([width - 2, height - 2])

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.done:
            info.update({
                'goal_index': self.goal_index
            })
        return obs, reward, done, info
