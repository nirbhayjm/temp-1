from gym_minigrid.minigrid import *
from ..minigrid_nav import *


class NarrowCorridor(NavGridEnv):
    """
    Simple grid world for point navigation
    """

    def __init__(
        self,
        size=4,
        max_steps=10,
        use_grid_in_state=False,
        normalize_agent_pos=False,
        obs_alpha=0.001,
        reward_scale=1.0,
        spawn='none',
        seed=123,
        reset_prob=1.0,
        perturb_prob=0.0,
        render_rgb = False,
        term_prob=0.0,
    ):
        # Make sure to set env attributes before super().__init__
        # assert spawn in ['none']
        # self.spawn = spawn
        assert size == 4, "Only one size (4) supported"

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


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        # NOTE: Do not generate surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        half_height = (self.grid.height+1)//2
        center = (self.grid.height)//2

        # Generate the wall in front of spawn location
        '''DEBUG'''
        # self.grid.vert_wall(1, 0, length=self.grid.height - 1)
        '''DEBUG'''
        # self.grid.horz_wall(1, center, length=width-1)

        """All walls"""
        self.grid.horz_wall(0, 0, length=4)
        self.grid.horz_wall(0, 1, length=2)
        self.grid.horz_wall(3, 1, length=1)

        self.grid.horz_wall(0, 3, length=1)
        self.grid.horz_wall(2, 3, length=1)
        self.start_pos = (1, 3)

        # Place the agent in the top-left corner
        # self.start_pos = (0, 0)
        # self.start_pos = (center, center)
        self.start_dir = 0

        # object_pos = [
        #     # [(1+width)//2, center],
        #     [width - 1, center - 1],
        #     [width - 1, center + 1],
        # ]

        # # '''DEBUG-SS'''
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

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.done:
            info.update({
                'goal_index': self.goal_index
            })
        return obs, reward, done, info
