from gym_minigrid.minigrid import *
from ..minigrid_nav import *


class RandomTwoGoals(NavGridEnv):
    """
    Simple grid world for point navigation
    """

    def __init__(
        self,
        size=5,
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
        render_rgb=False,
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
            config_seed=config_seed,
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

        # Generate the surrounding walls
        # NOTE: Do not generate surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        half_height = (self.grid.height+1)//2
        center = (self.grid.height)//2

        # Generate the wall in front of spawn location
        # self.grid.vert_wall(1, 1, length=half_height)
        # self.grid.horz_wall(1, center, length=width-1)

        # Place the agent in the top-left corner
        if self.spawn == 'center':
            self.start_pos = (center, center)
        else:
            # self.start_pos = (
            #     self._rand_int(0, width),
            #     self._rand_int(0, height)
            # )
            self.place_agent(rng=self.config_rng)

        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir
        self.start_dir = 0
        self.set_mission()

    def reset_agent_pos(self):
        if self.spawn == 'random':
            self.place_agent()
        self.agent_pos = self.start_pos
        self.start_dir = 0
        self.agent_dir = self.start_dir

    def reset(self):
        self.set_mission()
        obs = super().reset()
        return obs

    def set_mission(self):
        # corners = [
        #     [0, 0],
        #     [self.width - 1, 0],
        #     [0, self.height - 1],
        #     [self.width - 1, self.height - 1],
        # ]
        #
        # # object_pos = [
        # #     [self._rand_int(0, self.width), self._rand_int(0, self.height)],
        # #     [self._rand_int(0, self.width), self._rand_int(0, self.height)],
        # # ]
        #
        # _choice = self.np_random.choice(len(corners), 2, replace=False)
        # # object_pos = [corners[self._rand_int(0,4)]]
        # object_pos = [corners[idx] for idx in _choice]

        pos = np.array((
            self._rand_int(0, self.width),
            self._rand_int(0, self.height),
        ))
        # pos = np.array((self.width - 1, self.height - 1))
        # pos = np.array((0, 0))
        object_pos = [pos]

        # if hasattr(self, 'grid'):
        #     if hasattr(self, 'mission') and self.mission is not None:
        #         # Unset previous mission
        #         self.grid.set(*self.mission.astype('int'), None)
        #
        #     # Place two objects
        #     self.grid.set(*object_pos[0], NavObject())
        #     # self.grid.set(*object_pos[1], NavObject())

        # Flip a coin to decide which goal to select
        self.goal_index = 0
        # _rand = self._rand_int(0, 10)
        # self.goal_index = int(_rand < 3)
        # self.goal_index = int(self._rand_bool())
        self.mission = np.array(object_pos[self.goal_index]).astype('float')
        self.goal_hit = False

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.done:
            info.update({
                'goal_index': self.goal_index,
            })
        # Get a new random mission on success
        if self.success == True:
            self.set_mission()
            self.success = False
        return obs, reward, done, info
