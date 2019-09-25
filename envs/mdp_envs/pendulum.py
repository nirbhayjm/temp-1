# From: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

import math
from enum import IntEnum

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class MountainCarEnvCustom(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    class Actions(IntEnum):
        left = 0
        noop = 1
        right = 2

    def __init__(
        self,
        goal_velocity = 0,
        max_steps = 200,
        reset_on_done = True,
        spawn = 'random',
        reward_scale = 1.0,
        reset_prob = 1.0,
        render_rgb = False,
        end_on_goal = True,
        seed = 123,
    ):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.reset_on_done = reset_on_done
        self.max_steps = max_steps
        self.n_steps = 0
        self.reward_scale = reward_scale
        self.end_on_goal = end_on_goal
        assert spawn in ['random']
        self.spawn = spawn

        assert reset_prob >= 0.0 and reset_prob <= 1.0
        self.reset_prob = reset_prob
        # assert reset_prob >= 1.0
        self.first_reset = False

        self.render_rgb = render_rgb

        self.force=0.001
        self.gravity=0.0025

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.actions = MountainCarEnvCustom.Actions
        self.action_space = spaces.Discrete(3)

        # self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'pos-velocity': spaces.Box(
                low=self.low,
                high=self.high,
                dtype=np.float32,
            ),
            'mission': spaces.Box(
                low=0,
                high=2,
                shape=(2,),
                dtype=np.float32,
            ),
        })
        self.mission = np.zeros(2)

        self.seed(seed)

    def seed_config(self, *args, **kwargs):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = False
        if self.end_on_goal:
            done = bool(position >= self.goal_position \
                and velocity >= self.goal_velocity)

        reward = -1.0 * self.reward_scale

        self.n_steps += 1
        if self.n_steps >= self.max_steps:
            done = True

        self.state = (position, velocity)
        obs = {
            'pos-velocity': np.array(self.state).astype('float32'),
            'mission': self.mission.astype('float32'),
        }

        info = {'done': done}

        if self.render_rgb:
            # info['rgb_grid'] = self.render(mode='rgb_grid')
            info['rgb_grid'] = np.zeros((3, 3, 3))

        if self.reset_on_done:
            return obs, reward, done, info
        else:
            return obs, reward, False, info

    def reset(self):
        self.n_steps = 0

        do_reset = self.np_random.uniform(0.0, 1.0) <= self.reset_prob
        if do_reset or not self.first_reset:
            self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
            self.first_reset = True

        obs = {
            'pos-velocity': np.array(self.state).astype('float32'),
            'mission': self.mission.astype('float32'),
        }

        info = {'done': False}

        if self.render_rgb:
            # info['rgb_grid'] = self.render(mode='rgb_grid')
            info['rgb_grid'] = np.zeros((3, 3, 3))

        return obs, info

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='rgb_grid'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        tmp = self.viewer.render(return_rgb_array = mode=='rgb_array')
        import pdb; pdb.set_trace()

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_keys_to_action(self):
        return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
