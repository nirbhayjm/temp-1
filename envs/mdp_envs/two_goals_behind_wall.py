from gym_minigrid.minigrid import *
from ..minigrid_nav import *


class TwoGoalsBehindWall(NavGridEnv):

    def __init__(
        self,
        size=5,
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

        CORRIDOR_LEN = 1

        assert height % 2 == 1, "Only odd height supported"

        self.grid.wall_rect(0, 0, CORRIDOR_LEN, half_height-1)
        self.grid.wall_rect(0, half_height, CORRIDOR_LEN, half_height-1)

        # Generate the wall in front of spawn location
        # self.grid.vert_wall(1, 1, length=half_height)
        # self.grid.horz_wall(1, center, length=width-1)

        # Place the agent in the top-left corner
        self.start_pos = (0, center)
        # if self.spawn == 'center':
        #     self.start_pos = (center, center)
        # elif self.spawn == 'none':
        #     self.start_pos = (
        #         self._rand_int(0, width),
        #         self._rand_int(0, height)
        #     )
        self.start_dir = 0


        self.goal_index = 0
        self.mission = np.array([1, 1])
        self.set_mission()

    # def reset_agent_pos(self):
    #     self.place_agent()
    def reset_agent_pos(self):
        center = (self.grid.height)//2
        self.start_pos = (0, center)
        self.start_dir = 0

        self.agent_dir = self.start_dir
        self.agent_pos = self.start_pos

    def reset(self):
        self.set_mission()
        obs = super().reset()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.done:
            info.update({
                'goal_index': self.goal_index
            })
        return obs, reward, done, info

    def set_mission(self):
        # pos = np.array((
        #     self._rand_int(1, self.width),
        #     self._rand_int(0, self.height),
        # ))
        # object_pos = [pos]
        object_pos = [
            np.array((self.width - 1, 0)),
            np.array((self.width - 1, self.height - 1)),
        ]

        self.goal_index = self._rand_int(0, 2)

        self.mission = np.array(object_pos[self.goal_index]).astype('float')
        self.goal_hit = False
