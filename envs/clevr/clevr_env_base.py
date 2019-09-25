import os
import json
import numpy as np
import math
from collections import defaultdict
from envs.clevr.controller import Controller

import gym

class ClevrEnvBase(gym.Env):

    def __init__(
            self,
            env_base_dir,
            pov_imgs_path,
            env_list,
            num_agents=1,
            num_head_turns=8,
            num_body_turns=4,
            action_space_id=0,
            grid_size=5,
            f_cell_occ_map='cell_occ_map.npy',
            f_cell_attr_map='cell_attr_map.npy',
            env_dim=6,
            viz_dir='data/viz',
            img_dim=64,
            enable_viz=False,
            spawn_curriculum='none',
            debug=False,
            observability='full'):

        self.env_base_dir = env_base_dir
        self.pov_imgs_path = pov_imgs_path
        self.env_list = env_list
        self.num_agents = num_agents
        self.num_head_turns = num_head_turns
        self.num_body_turns = num_body_turns
        self.grid_size = grid_size
        self.env_dim = env_dim
        self._ctrl = None
        self.viz_dir = viz_dir
        self.f_cell_occ_map = f_cell_occ_map
        self.f_cell_attr_map = f_cell_attr_map
        self.img_dim = img_dim
        self._stay_action = [0 for _ in range(self.num_agents)]
        self.env_dir_sfx = 'e0'
        self.enable_viz = enable_viz
        self.action_space_id = action_space_id
        self.spawn_curriculum = spawn_curriculum
        self.debug = debug
        self.observability = observability

        # [NOTE] : 'args.action_space' determines which action space is used
        # for navigation

        #if self.observations not in [1, 3, 5]:
        #    raise ValueError("Uknown obs: {}".format(self.observations))
        # Direction vectors
        self.directions = {
                'left' : [-0.6563112735748291, -0.7544902563095093],
                'right' : [0.6563112735748291, 0.7544902563095093],
                'front' : [0.754490315914154, -0.6563112735748291],
                'back' : [-0.754490315914154, 0.6563112735748291]
        }

        self._ctrl = None
        self.loaded_viewpoints = None

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        """
        Reset the environment.
        Sample a new environment, and agent configurations.
        """
        self._base_reset()

    def step(self, actions):
        """
        Take a step defined by actions for all the agents
        """

        # Step can only be taken if Environment has been reset before
        assert self._ctrl is not None, \
                "No Step can be taken before Environemnt has been reset."

        self.curr_actions = actions
        # Take a step
        self._ctrl.step(actions)

    def _base_reset(self):
        """
        Reset agents and environment.
        """
        # Set the environment index
        self.env_idx = self._rand_env_idx()

        # env_id is path to the directory with environment file
        env_id = os.path.join(
                self.env_base_dir,
                self.env_list[self.env_idx],
                self.env_dir_sfx)

        if self.debug:

            print("-----------------------------------")
            print("ENV RESET TO: {}".format(self.env_list[self.env_idx]))
            print("-----------------------------------")

        if self._ctrl is None or self._ctrl.env_id != env_id:
            self._load_env_data(env_id)

        if self.spawn_curriculum == 'none':
            self.agent_cfgs = self._rand_agent_cfgs()
        elif self.spawn_curriculum == 'center':
            self.agent_cfgs = self._center_spawn_cfgs()
        # elif self.spawn_curriculum == 'one-step':
        #     self.agent_cfgs = self._get_spawn_locs()
        else:
            raise ValueError


        if self._ctrl is not None and self._ctrl.env_id == env_id:

            if self.debug:
                print("-----------------------------------")
                print("ENV NOT LOADED.")
                print("-----------------------------------")
            # Environment remains the same. Only agents' configuration change
            self._ctrl.reset_agent(self.agent_cfgs)

        else:

            if self.debug:
                print("-----------------------------------")
                print("ENV WILL BE LOADED.")
                print("-----------------------------------")

            # A new environment has to be loaded; create a new Controller
            ctrl = Controller(
                    env_id=env_id,
                    agent_cfgs=self.agent_cfgs,
                    cell_attr_map=self.cell_attr_map,
                    cell_occ_map=self.cell_occ_map,
                    pov_imgs_path=self.pov_imgs_path,
                    num_agents=self.num_agents,
                    num_head_turns=self.num_head_turns,
                    num_body_turns=self.num_body_turns,
                    grid_size=self.grid_size,
                    env_dim=self.env_dim,
                    viz_dir=self.viz_dir,
                    img_dim=self.img_dim,
                    enable_viz=self.enable_viz,
                    action_space_id=self.action_space_id,
                    observability=self.observability)

            self._ctrl = ctrl
            if self.loaded_viewpoints is not None:
                self._ctrl._load_obs(self.loaded_viewpoints)

    def viz(self):
        """
        Visualize the environment and agent's position in it.
        """
        self._ctrl.viz()

    def get_agent_cfg(self):
        """
        Get agent's current configuration
        """
        #if self.observations == 1:
        #    cfgs = [np.array([agent.row, agent.col, agent.th, agent.th]) for\
        #        agent in self._ctrl.agents]
        #elif self.observations == 3:
        #    cfgs = [np.array([agent.row, agent.col, agent.tb, agent.tb]) for\
        #        agent in self._ctrl.agents]
        #elif self.observations == 5:
        #    cfgs = [np.array([agent.row, agent.col, agent.tb, agent.tb]) for\
        #        agent in self._ctrl.agents]
        #else:
        #    raise ValueError
        if self.action_space_id == 0:
            cfgs = [np.array([agent.row, agent.col, agent.tb, agent.tb]) for\
                agent in self._ctrl.agents]
        else:
            raise NotImplementedError
        return np.stack(cfgs, 0)

    def get_agent_pov(self):
        """
        Get agent's current POV image
        """
        return np.stack([self._ctrl._agent_cfg_to_img(agent) for agent in \
            self._ctrl.agents], 0)

    def _rand_agent_cfgs(self):
        """
        Sample a random joint cfgs for all the agents.
        """

        A = self.num_agents
        N = self.grid_size
        HT = self.num_head_turns
        BT = self.num_body_turns

        rand_cfgs = np.empty((A,4),dtype=np.int64)

        rand_pos = np.random.randint(N,size=(A,2))
        for a in range(A):
            while True:
                a_i = np.random.randint(N)
                a_j = np.random.randint(N)

                if self.cell_occ_map[a_i][a_j] == 0:
                    rand_cfgs[a][0] = a_i
                    rand_cfgs[a][1] = a_j
                    break

        rand_head_orient = np.random.randint(HT,size=A)
        rand_body_orient = np.random.randint(BT,size=A)


        rand_cfgs[:,:2] = rand_pos
        rand_cfgs[:,2] = rand_body_orient
        rand_cfgs[:,3] = rand_head_orient

        return rand_cfgs

    def _center_spawn_cfgs(self):
        """
        Sample agents at the center of the grid.
        """

        A = self.num_agents
        N = self.grid_size
        HT = self.num_head_turns
        BT = self.num_body_turns

        rand_cfgs = np.empty((A,4), dtype=np.int64)

        available_cells = np.where(self.cell_occ_map == 0)
        _x = available_cells[0]
        _y = available_cells[1]

        dist_from_center = (_x - (0.5 * N))**2 + (_y - (0.5 * N))**2

        _index = dist_from_center.argmin()

        x_center = _x[_index]
        y_center = _y[_index]

        rand_cfgs[:, 0] = x_center
        rand_cfgs[:, 1] = y_center

        rand_head_orient = np.random.randint(HT,size=A)
        rand_body_orient = np.random.randint(BT,size=A)

        rand_cfgs[:,2] = rand_body_orient
        rand_cfgs[:,3] = rand_head_orient

        return rand_cfgs

    def _get_spawn_locs(self):
        raise NotImplementedError

    def _rand_env_idx(self):
        """
        Sample a random environment to initialize the Controller
        """

        return np.random.randint(len(self.env_list))

    def _validate_agent_cfgs(self):
        """
        Validate configurations of all the agents.
        """
        assert (self.agent_cfgs.shape == (self.num_agents,3)) # Right shape

        val_x = np.logical_and(
                self.agent_cfgs[:,0] < self.grid_size,
                self.gent_cfgs[:,0] >= 0)

        val_y = np.logical_and(
                self.agent_cfgs[:,1] < self.grid_size,
                self.agent_cfgs[:,1] >= 0)

        assert np.all(val_x) and np.all(val_y), "Agents' grid positions are misspecified."

        val_t = np.logical_and(
                self.agent_cfgs[:,2] >= 0,
               self.agent_cfgs[:,2] < self.num_head_turns)

        assert np.all(val_t) , "Agents' orientations are misspecified."

    def _read_json(self,fname):
        """
        Read JSON data from file.
        """

        with open(fname) as f:
            data = json.load(f)

        return data

    def _load_env_data(self,env_id):
        """
        Load data to be passed to Controller class
        """

        f_cell_occ_map = os.path.join(env_id,self.f_cell_occ_map)
        f_cell_attr_map = os.path.join(env_id,self.f_cell_attr_map)

        self.cell_occ_map = np.load(f_cell_occ_map)
        self.cell_attr_map = np.load(f_cell_attr_map)

    def _orientation_vectors(self):
        """
        Return Ax2 dimensional numpy array representing a unit vector
        for directions each of the agent is facing.
        """

        agent_orientations = np.empty((self.num_agents,2),dtype=np.float)

        for a_idx, a in enumerate(self._ctrl.agents):
            theta = a.th*2*math.pi/self.num_head_turns
            agent_orientations[a_idx] = [-1*math.sin(theta),math.cos(theta)]

        return agent_orientations
