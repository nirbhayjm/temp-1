from gym_minigrid.minigrid import *
from ..minigrid_nav import *


class PillarGrid(NavGridEnv):
    """
    Simple grid world for point navigation
    """

    def __init__(
        self,
        size=5,
        max_steps=10,
        use_grid_in_state=False,
        normalize_agent_pos=False,
        obstacle_type='wall',
        obs_alpha=0.001,
        reward_scale=1.0,
        spawn='none',
        seed=123,
        reset_prob=1.0,
        perturb_prob=0.0,
        render_rgb=False,
        term_prob=0.0,
    ):
        self.spawn = spawn
        self.obstacle_type = obstacle_type
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
            # 'mission': spaces.MultiDiscrete([size, size]),
            'mission': spaces.Box(
                low=0.0,
                high=size,
                shape=(2,),
                dtype='float',
            ),
        })
        self.observation_space = spaces.Dict(new_spaces)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        half_height = (self.grid.height+1)//2
        center = (self.grid.height)//2

        # Generate pillar grid
        for i in range(0, width, 2):
            for j in range(0, width, 2):
                if self.obstacle_type == 'wall':
                    self.grid.set(i, j, Wall())
                elif self.obstacle_type == 'lava':
                    self.grid.set(i, j, Lava())
                else:
                    raise ValueError

        self.mission = np.array([0, 0], dtype='int64')
        # Place the agent in the top-left corner
        if self.spawn == 'center':
            self.start_pos = (center, center)
        elif self.spawn == 'none':
            # self.start_pos = (
            #     self._rand_int(0, width),
            #     self._rand_int(0, height)
            # )
            self.place_agent()
        self.agent_pos = self.start_pos
        self.start_dir = 0

    def reset_agent_pos(self):
        if self.spawn == 'center':
            self.start_pos = (1, 1)
        elif self.spawn == 'none':
            self.place_agent()
        self.start_dir = 0

        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir
