from gym_minigrid.minigrid import *
# from gym_minigrid.envs import MultiRoomEnv
# from ..minigrid_nav import *
from ..pomdp_envs.multiroom import MultiroomWrapper


CELL_PIXELS = 8

class MultiroomMDP(MultiroomWrapper):
    """
    Simple grid world for point navigation
    """

    def __init__(
        self,
        max_steps=10,
        obs_alpha=0.001,
        render_rgb=False,
        term_prob=0.0,
        reset_prob=1.0,
        seed=123,
    ):
        # assert minNumRooms > 0
        # assert maxNumRooms >= minNumRooms
        # assert maxRoomSize >= 4
        #
        # self.minNumRooms = minNumRooms
        # self.maxNumRooms = maxNumRooms
        # self.maxRoomSize = maxRoomSize
        #
        # self.rooms = []

        # self.static_grid = True
        # self.grid_generated = False

        # super().__init__(
        #     minNumRooms=5,
        #     maxNumRooms=5,
        #     maxRoomSize=25,
        # )
        #
        self.encoding_range = len(OBJECT_TO_IDX)
        super().__init__(
            max_steps=max_steps,
            obs_alpha=obs_alpha,
            render_rgb=render_rgb,
            term_prob=term_prob,
            reset_prob=reset_prob,
            seed=seed,
        )
        new_spaces = self.observation_space.spaces
        new_spaces.update({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3, self.encoding_range),
                dtype='float',
            ),
            'direction': spaces.Box(
                low=0,
                high=3,
                shape=(4,),
                dtype='float',
            ),
            'mission': spaces.Box(
                low=0,
                high=1,
                shape=(10,),
                dtype='float',
            ),
        })
        self.observation_space = spaces.Dict(new_spaces)

        # NavGridEnv.__init__(
        #     self,
        #     grid_size=maxRoomSize,
        #     width=None,
        #     height=None,
        #     max_steps=max_steps,
        #     see_through_walls=False,
        #     use_grid_in_state=use_grid_in_state,
        #     normalize_agent_pos=normalize_agent_pos,
        #     obs_alpha=obs_alpha,
        #     reward_scale=reward_scale,
        #     reset_start=reset_start,
        #     term_prob=term_prob,
        #     render_rgb=render_rgb,
        #     seed=seed,
        # )

        # new_spaces = self.observation_space.spaces
        # new_spaces.update({
        #     'mission': spaces.MultiDiscrete([maxRoomSize, maxRoomSize]),
        # })
        # self.observation_space = spaces.Dict(new_spaces)

    def encode_grid(self):
        orig_image = self.grid.encode(vis_mask=None)
        one_hot_img = np.eye(self.encoding_range)[orig_image]
        return one_hot_img.astype(np.float32)

    def gen_obs(self):
        """
        Generate the agent's view (fully observable grid)
        """
        # image = self.grid.encode(vis_mask=None)
        agent_pos = np.array(self.agent_pos)

        assert hasattr(self, 'mission'), \
            "environments must define a mission"

        obs = {
            'image': self.encode_grid(),
            'direction': np.eye(4)[self.agent_dir],
            'agent_pos': agent_pos.astype(np.float32),
            'mission': self.mission,
        }

        for key in obs.keys():
            if hasattr(obs[key], 'dtype') and obs[key].dtype == 'float':
                # Use float32 instead of float64
                obs[key] = obs[key].astype('float32')
        return obs

    # def _gen_grid(self, width, height):
    #     if not self.static_grid or not self.grid_generated:
    #         MultiRoomEnv._gen_grid(self, width, height)
    #         self.grid_generated = True
    #     # self.mission = np.array(object_pos[self.goal_index],
    #     #     dtype='int64')
    #     self.mission = self.goal_pos

    # def step(self, action):
    #     obs, reward, done, info = super().step(action)
    #     if self.done:
    #         info.update({
    #             'goal_index': 0,
    #         })
    #     return obs, reward, done, info
