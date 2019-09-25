from gym_minigrid.minigrid import *
from ..minigrid_nav import *

import numpy as np

class TwoRoomCorridor(NavGridEnv):
    """
    Two rooms connected by a long corridor. Size of the rooms and that of
    corridor are configurable.
    """

    def __init__(
        self,
        size,
        corridor_len,
        agent_start_room='left',
        max_steps=10,
        use_grid_in_state=False,
        normalize_agent_pos=False,
        obs_alpha=0.001,
        reward_scale=1.0,
        spawn='random',
        seed=123,
        config_seed=123,
        reset_prob=1.0,
        perturb_prob=0.0,
        render_rgb = False,
        term_prob=0.0,
    ):
        self.corridor_len = corridor_len
        self.agent_start_room = agent_start_room
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
                config_seed=config_seed,
            )

        self.corridor_len = corridor_len
        self.agent_start_room = agent_start_room

        new_spaces = self.observation_space.spaces
        new_spaces.update({
            'mission': spaces.MultiDiscrete([size, size]),
        })
        self.observation_space = spaces.Dict(new_spaces)

    def wall_rect_filled(self, x, y, w, h):
        for i in range(0, w):
            for j in range(0, h):
                self.grid.set(x + i, y + j, Wall())

    def _gen_grid(self, width, height):

        # Create empty grid
        self.grid = Grid(width, height)

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

        # if self.agent_start_room == 'left':
        #     # start_x = np.random.choice(np.arange(1, corr_x))
        #     start_x = 1
        # elif self.agent_start_room == 'right':
        #     # start_x = np.random.choice(np.arange(corr_x + self.corridor_len, width - 1))
        #     start_x = width - 1
        # else:
        #     raise ValueError

        # start_y = np.random.choice(np.arange(1, height - 1))
        # start_y = 1

        # print(start_x, start_y)
        if self.spawn == 'center':
            self.start_pos = (self.width // 2, corr_y)
        # elif self.spawn == 'fixed':
        #     self.start_pos = (1, 1)
        else:
            self.place_agent(rng=self.config_rng)
        self.start_dir = 0
        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir

        center = (self.grid.height)//2
        object_pos = [
            # [(1+width)//2, center],
            [width - 1, center - 1],
            [width - 1, center + 1],
        ]
        # # '''DEBUG-SS'''

        # Place two objects
        # self.grid.set(*object_pos[0], NavObject())
        # self.grid.set(*object_pos[1], NavObject())

        # Flip a coin to decide which goal to select
        # self.goal_index = 0
        # _rand = self._rand_int(0, 10)
        # self.goal_index = int(_rand < 3)
        self.goal_index = int(self._rand_bool())
        self.mission = np.array(object_pos[self.goal_index],
            dtype='int64')

    def reset_agent_pos(self):
        if self.spawn == 'random':
            self.place_agent()
        self.agent_pos = self.start_pos
        self.start_dir = 0
        self.agent_dir = self.start_dir
