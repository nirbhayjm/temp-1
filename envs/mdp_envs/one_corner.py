from gym_minigrid.minigrid import *
from ..minigrid_nav import *


class OneCornerGoal(NavGridEnv):
    """
    Simple grid world for point navigation
    """

    def __init__(
        self,
        size=5,
        max_steps=10,
        use_grid_in_state=False,
        normalize_agent_pos=False,
        obs_alpha=0.0,
        reward_scale=1.0,
        reset_prob=1.0,
        perturb_prob=0.0,
        term_prob=0.0,
        spawn='none',
        config_seed=0,
        render_rgb=False,
        static_grid=True,
        seed=123,
    ):
        # Make sure to set env attributes before super().__init__
        self.spawn = spawn

        super().__init__(
            grid_size=size,
            width=None,
            height=None,
            max_steps=max_steps,
            see_through_walls=False,
            use_grid_in_state=use_grid_in_state,
            normalize_agent_pos=normalize_agent_pos,
            reset_prob=reset_prob,
            term_prob=term_prob,
            perturb_prob=perturb_prob,
            render_rgb=render_rgb,
            obs_alpha=obs_alpha,
            reward_scale=reward_scale,
            static_grid=static_grid,
            config_seed=config_seed,
            seed=seed,
        )

        # Update obs space after super().__init__
        new_spaces = self.observation_space.spaces
        new_spaces.update({
            'mission': spaces.MultiDiscrete([size, size]),
        })
        self.observation_space = spaces.Dict(new_spaces)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        half_height = (self.grid.height+1)//2
        center = (self.grid.height)//2

        # Place the agent in the top-left corner
        if self.spawn == 'none':
            self.start_pos = (self._rand_int(0, width),
                              self._rand_int(0, height))
        elif self.spawn == 'center':
            self.start_pos = (center, center)
        else:
            raise ValueError("{}".format(self.spawn))
        self.start_dir = 0

        corners = [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
        ]

        # # Place one object at every corner
        # self.grid.set(*corners[0], NavObject())
        # self.grid.set(*corners[1], NavObject())
        # self.grid.set(*corners[2], NavObject())
        # self.grid.set(*corners[3], NavObject())


        _choice = np.random.choice(len(corners), replace=False)
        object_pos = corners[_choice]

        # Flip a coin to decide which goal to select
        self.goal_index = 0
        # _rand = self._rand_int(0, 10)
        self.mission = np.array(object_pos)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.done:
            info.update({
                'goal_index': self.goal_index
            })
        return obs, reward, done, info
