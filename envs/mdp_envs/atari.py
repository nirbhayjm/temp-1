# From: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

import math
from enum import IntEnum

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class AtariEnvCustom(AtariEnv):

    # class Actions(IntEnum):
    #     left = 0
    #     noop = 1
    #     right = 2

    def __init__(
        self,
        game='freeway',
        atari_obs_type='ram',
        atari_mode=None,
        atari_difficulty=None,
        frameskip=(2, 5),
        repeat_action_probability=0.,
        full_action_space=False,

        # goal_velocity = 0,
        max_steps = 200,
        reset_on_done = True,
        spawn = 'random',
        reward_scale = 1.0,
        reset_prob = 1.0,
        render_rgb = False,
        end_on_goal = True,
        seed = 123,
    ):
        super().__init__(
            game=game,
            obs_type=atari_obs_type,
            mode=atari_mode,
            difficulty=atari_difficulty,
            frameskip=frameskip,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
        )
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
