import numpy as np
from .empty import EmptyGridNav

class RGBEmptyGridNav(EmptyGridNav):

    def __init__(self,
                 size=5,
                 max_steps=10):

        self.color_map = {
            0 : [255,255,255], # empty
            1 : [47,79,79], # Wall -> grey
            2 : [0, 128, 0], # ball -> green
            3 : [0, 0, 0]
        }

        self.random_push_prob = random_push_prob

        super().__init__(size=size, max_steps=max_steps)

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
