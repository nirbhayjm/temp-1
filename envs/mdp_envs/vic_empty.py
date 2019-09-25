from gym_minigrid.minigrid import *
from ..minigrid_nav import *

import numpy as np

DEBUG = False

class VICEmpty(NavGridEnv):
    """
    Simple grid world for point navigation
    """

    def __init__(self,
            size=5,
            max_steps=10,
            random_push_prob=0.2,
            alive_prob=0.95,
            use_grid_in_state=True):

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            use_grid_in_state=True)

        self.random_push_prob = random_push_prob
        self.alive_prob = alive_prob
        self.die_prob = 1 - alive_prob

        self.action_list = [
               self.actions.right,
               self.actions.down,
               self.actions.left,
               self.actions.up,
               self.actions.done
        ]

    def step(self, action):

        if action in self.move_actions:
            if np.random.random() <= self.random_push_prob:

                if DEBUG:
                    print("Agent Randomly Pushed by the wind")
                aidx = np.random.randint(len(self.move_actions))
                action = self.move_actions[aidx]

        obs, reward, _, info = super().step(action)

        if self.step_count >= self.max_steps - 1:
            self.done = True
        else:
            # done action doesn't end the task
            # essentially a stay action
            self.done = False

        # In VIC world, the agent can also die randomly
        if np.random.random() <= self.die_prob:

            if DEBUG:
                print("Environment Killed the agent.")
            self.done = True

        info['done'] = self.done

        return obs, reward, False, info


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, NavObject())

        # Define unmissioned environment as empty numpy array
        self.mission = np.array([0,0])

class RGBVICEmpty(VICEmpty):

    def __init__(self,
                 size=5,
                 max_steps=10,
                 random_push_prob=0.2,
                 alive_prob=0.95,
                 use_grid_in_state=True):

        self.color_map = {
            0 : [255,255,255], # empty
            1 : [47,79,79], # Wall -> grey
            2 : [0, 128, 0], # ball -> green
            3 : [0, 0, 0] # Color
        }

        self.random_push_prob = random_push_prob
        self.alive_prob = alive_prob
        self.die_prob = 1 - alive_prob

        super().__init__(
                size=size,
                max_steps=max_steps,
                random_push_prob=random_push_prob,
                alive_prob=alive_prob,
                use_grid_in_state=use_grid_in_state)

    def reset(self):
        obs = super().reset()

        obs = self._rgbize(obs)

        return obs

    def step(self, action):

        obs, reward, done, info = super().step(action)

        obs = self._rgbize(obs)

        return obs, reward, done, info

    def _rgbize(self, obs):
        # Extract the image
        img = obs['image'].copy()
        new_img = np.zeros(img.shape[:2] + (3, ), dtype=int)
        for k, v in self.color_map.items():
            new_img[img[:,:,0] == k] = v

        obs.update({'image' : new_img })

        return obs
