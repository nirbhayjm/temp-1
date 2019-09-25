from gym_minigrid.minigrid import *
from ..minigrid_nav import *

import numpy as np

class FourRooms(NavGridEnv):
    """
    4 rooms connected by a single opening
    """

    def __init__(
        self,
        size,
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

        new_spaces = self.observation_space.spaces
        new_spaces.update({
            'mission': spaces.MultiDiscrete([size, size]),
        })
        self.observation_space = spaces.Dict(new_spaces)

    def _gen_grid(self, width, height):

        # Create empty grid
        self.grid = Grid(width, height)

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

        # # For now, start only at the bottom left room:
        # start_x = np.random.choice(np.arange(1, cx))
        # start_y = np.random.choice(np.arange(1, cy))
        # self.start_pos = (start_x, start_y)
        if self.spawn == 'random':
            self.place_agent(rng=self.config_rng)
        else:
            self.start_pos = (1, 1)
        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir
        self.start_dir = 0

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
