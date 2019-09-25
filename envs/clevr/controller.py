import os
import json
import h5py
import math
import numpy as np
# from PIL import Image

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Arrow
from matplotlib.image import AxesImage

from matplotlib.collections import PatchCollection

from envs.clevr.agent import Agent

class Controller:

    def __init__(
            self,
            # Same as the dir with env files
            env_id,
            agent_cfgs,
            cell_attr_map,
            cell_occ_map,
            pov_imgs_path,
            num_agents,
            num_head_turns,
            num_body_turns,
            grid_size,
            env_dim,
            viz_dir,
            img_dim,
            enable_viz,
            action_space_id,
            observability):

        # Environment Specific members
        self.env_id = env_id

        self.cell_attr_map = cell_attr_map
        self.cell_occ_map = cell_occ_map
        self.pov_imgs_path = pov_imgs_path
        self.env_dim = env_dim
        # File where 2D visualization of environment and agent is saved.
        self.viz_dir = viz_dir

        # Number of times self.step() is called.
        # Used to name the visualization of environment after that step
        self.step_count = 0
        self.num_agents = num_agents

        # Discretization details
        self.grid_size = grid_size
        self.num_head_turns = num_head_turns
        self.num_body_turns = num_body_turns
        self.observability = observability

        # Create Agents
        self.agents = self._create_agents(agent_cfgs)

        self.img_dim = img_dim

        self.enable_viz = enable_viz
        self.action_space_id = action_space_id
        # matplotlib objects used for visualization
        # This is pretty hacky. There has to be a better way to do viz
        self._fig = None
        self._ogrid = None
        self._cells = None
        self._apov = None

        self._fillcolor = 'blue'
        self._emptycolor = 'white'
        self._agentownarrow = 'solid'
        self._agentotherarrow = 'dotted'
        self._agentownarrowcolor = 'black'
        self._agentotherarrowcolor = 'gray'
        self._agentcelllinestyle = 'dashed'
        self._agentcellcolor = 'red'
        # Setup the visualization

        if self.enable_viz:
            self._setup_viz()

        self.viewpoints = None
        # print("START LOADING: {}".format(env_id))
        # self._load_obs()
        # print("DONE LOADING: {}".format(env_id))


    def step(self,actions):
        """
        Update the agent configs as specified by the actions.
        """

        assert len(actions) == self.num_agents, \
                "Specify action for each agent"

        # First assert if actions are valid
        #for a in actions:
        #    assert (a in range(6)), "Actions have to be integers in [0-5]"
        if self.action_space_id == 0:
            mov = self._move_cardinal
        else:
            raise NotImplementedError
        # Use actions to move agents
        for a_idx, a in enumerate(self.agents):
            mov(a, actions[a_idx])
        self.step_count += 1


    def reset_agent(self,agent_cfgs):
        """
        Reset the agent configs in the same environment
        """
        self.step_count = 0
        self.num_agents = agent_cfgs.shape[0]
        self.agents = self._create_agents(agent_cfgs)

    def viz(self):
        """
        Visualize the 2D occupancy map for the scene and agent viewpoint for
        each agent.
        """

        #------------------------------#
        # Visualize the occupancy grid #
        #------------------------------#

        # Take the agent_configs, update the cells where agents are located,
        # refill the old agent locations with emptycolor.

        # If agent's movement has made a cell empty, fill it with emptycolor

        N = self.grid_size
        A = self.num_agents

        for a in range(A):
            # Fill a cell with empty color if it was vacated by a agent.
            self._fill_empty_cells(a)

            # Add agent cells for all agents. Every agent keeps information
            # of all other agents. Agent's own cell is self._agentowncolor
            # while cells of other agents are self._agentothercolor.

            self._fill_agent_cells(a)

            # Update PatchCollection for self._ogrid[a] axes
            pc = PatchCollection(self._cells[a],match_original=True)
            self._ogrid[a].collections[0].remove()
            self._ogrid[a].add_collection(pc)

            #---------------------------#
            # Visualize agent's POV     #
            #---------------------------#


            # Make sure there is only one AxesImage on self._apov[a] axes
            ax_img = None
            for obj in self._apov[a].get_children():
                if isinstance(obj,AxesImage):
                    ax_img = obj
                    break
            if ax_img is not None:
                ax_img.remove()

            # Add the new image to self._apov
            self._apov[a].imshow(self._agent_cfg_to_img(self.agents[a]))
            # Save the visualization
            self._save_fig(a)


    def _setup_viz(self):
        """
        Create a figure and 2 axes for occupancy map and agent's pov
        Agent's position is never added to the self._fig
        """

        A = self.num_agents
        self._fig = [None for _ in range(A)]
        self._ogrid = [None for _ in range(A)]
        self._apov = [None for _ in range(A)]
        self._cells = [None for _ in range(A)]

        for a in range(A):
            self._setup_agent_viz(a)


    def _load_obs(self, obs=None):
        """
        Load all the agent viewpoints in the cache.
        """
        if obs is None:
            print("[WARN]: Loading obs from disk.")
            raise NotImplementedError
            with h5py.File(self.pov_imgs_path, 'r') as hfile:
                env_index = int(os.path.dirname(self.env_id)[-6:])
                self.viewpoints = np.array(
                    hfile['data_mats'][env_index], dtype='float')
                self.viewpoints = self.viewpoints / 255.0

        else:
            self.viewpoints = obs

        # N = self.grid_size
        # T = self.num_head_turns
        # D = self.img_dim
        #
        # self.viewpoints = np.empty((N,N,T,D,D,3),dtype=np.float64)
        #
        # for i in range(N):
        #     for j in range(N):
        #         for t in range(T):
        #             fname = '{0}_{1}_{2}.png'.format(i,j,t)
        #             fpath = os.path.join(self.env_id,'img',fname)
        #             im_pil = Image.open(fpath)
        #             im_np = np.array(im_pil)/255.0
        #             self.viewpoints[i,j,t] = im_np


    def _setup_agent_viz(self,agent_id,figsize=(10,10)):
        """
        Setup visualization for a single agent.
        """


        fig = plt.figure(figsize=figsize)
        apov_size = [0.1,0.1,0.5,0.5]
        grid_size = [0.62,0.62,0.35,0.35]

        # Axes for images from agents POV
        apov = fig.add_axes(apov_size)
        # Axes for occupancy grid
        ogrid = fig.add_axes(grid_size)

        apov.get_xaxis().set_visible(False)
        apov.get_yaxis().set_visible(False)

        ogrid.get_xaxis().set_visible(False)
        ogrid.get_yaxis().set_visible(False)

        self._fig[agent_id] = fig
        self._ogrid[agent_id] = ogrid
        self._apov[agent_id] = apov
        self._add_cell_occ_map(agent_id)

    def _fill_empty_cells(self,agent_id):
        """
        Fill in the self._cells array for agent_id with empty cells
        """

        N = self.grid_size
        A = self.num_agents

        for i in range(N):
            for j in range(N):
                agent_occ = False
                if self.cell_occ_map[i][j] == 0:
                    # Check if it's occupied by an agent
                    for a_idx,a in enumerate(self.agents):
                        a_i = a.row; a_j = a.col
                        if i == a_i and j == a_j:
                            agent_occ = True

                    if not agent_occ:
                        # Cell is not occupied by any agent
                        cell_idx = i*N + j
                        cell = self._cells[agent_id][cell_idx]
                        x,y = cell.get_x(),cell.get_y()
                        dx,dy = cell.get_width(),cell.get_height()

                        self._cells[agent_id][cell_idx] = Rectangle(
                                                (x,y),dx,dy,
                                                facecolor=self._emptycolor,
                                                edgecolor='black')


    def _fill_agent_cells(self,agent_id):
        """
        Add agent cells for all the agents from agent_id's perspective.
        """

        N = self.grid_size

        for a_idx,a in enumerate(self.agents):
            a_i,a_j = a.row,a.col
            cell_idx = a_i*N + a_j
            cell  = self._cells[agent_id][cell_idx]
            x,y = cell.get_x(),cell.get_y()
            dx,dy = cell.get_width(),cell.get_height()

            if a_idx == agent_id:
                arrowcolor = self._agentownarrowcolor
                linestyle = self._agentcelllinestyle
            else:
                arrowcolor = self._agentotherarrowcolor
                linestyle = 'solid'

            self._cells[agent_id][cell_idx] = Rectangle(
                                            (x,y),dx,dy,
                                            facecolor=self._agentcellcolor,
                                            edgecolor='black',
                                            linestyle=linestyle)

            # Update the arrow patch for this agent
            arrow_idx = N*N + a_idx
            a_x  = x + dx/2
            a_y = y + dy/2

            a_r = dx/2.5

            a_theta = math.pi/2 + 2*math.pi*a.th/self.num_head_turns

            a_dx = a_r*math.cos(a_theta)
            a_dy = a_r*math.sin(a_theta)

            self._cells[agent_id][arrow_idx] = Arrow (
                                        a_x,a_y,a_dx,a_dy,
                                        width=0.02,color=arrowcolor)



    def _add_cell_occ_map(self,agent_id):
        """
        Initialize the agent ogrid with cell_occ_map for the current environment.
        """
        N = self.grid_size
        A = self.num_agents
        cells = []

        dx = 1.0/N; dy = 1.0/N

        for i in range(N):
            y = i*dy
            x = 0
            for j in range(N):

                if self.cell_occ_map[i][j] == 0:
                    facecolor = self._emptycolor
                else:
                    facecolor = self._fillcolor

                cells.append(
                        Rectangle(
                            (x,y),dx,dy,facecolor=facecolor,
                            edgecolor='black'))
                x += dx
        # Create dummy cells for arrows. One arrow needed for each agent
        for _ in range(A):
            cells.append(None)
        # Create a PatchCollection
        self._cells[agent_id] = cells
        pc = PatchCollection(cells[:N*N],match_original=True)
        self._ogrid[agent_id].add_collection(pc)

    def _save_fig(self,agent_id):
        """
        Save self._fig as <self.viz_dir>/<agent_id>_<self.step_count>.png
        """

        outdir = os.path.join(self.viz_dir,'{}'.format(agent_id))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outfpath = os.path.join(outdir,'{0:03d}.png'.format(self.step_count))

        self._fig[agent_id].savefig(outfpath)


    def _create_agents(self,agent_cfgs):
        """
        Create agents specified by agent_cfgs
        """

        agents = []
        for idx  in range(self.num_agents):
            cfg = agent_cfgs[idx]
            a = Agent(cfg[0],cfg[1],cfg[2],cfg[3])
            agents.append(a)

        return agents

    def _agent_cfg_to_img(self,agent, front_only=False):
        """
        Return a self.observations*H*D*3 np array from agent POV.
        """
        if self.observability == 'partial-single':

            obs =  self.viewpoints[agent.row,agent.col,agent.th][None,...]

        elif self.observability == 'partial-triplet':

            idx_map = {
                0 : [-1,0,1],
                1 : [1,2,3],
                2 : [3,4,5],
                3 : [5,6,7]}

            if not front_only:
                obs = torch.stack([self.viewpoints[agent.row,agent.col,idx] \
                        for idx in idx_map[agent.tb]], 0)

            else: # Return only front image
                obs = self.viewpoints[agent.row,agent.col,idx_map[agent.tb][1]]

        else:
            raise ValueError
        return obs

    def _cfg_to_img(self, row, col, th, tb):
        """
        Arbitrary cfg POV
        """
        if self.obervability == 'partial-single':

            return self.viewpoints[row,col,th]

        elif self.observability == 'partial-triplet':

            idx_map = {
                0 : [-1,0,1],
                1 : [1,2,3],
                2 : [3,4,5],
                3 : [5,6,7]}
            return self.viewpoints[row,col,idx_map[tb]]

        else:
            raise ValueError

    def _fetch_obs(self, agent=None):
        """
        Read the images fron each agent's viewpoint and return after
        concatenation.
        """

        if type(agent) == int:
            obs = torch.stack(self._agent_cfg_to_img(self.agents[agent]),0)
        elif type(agent) == list:
            obs = torch.stack(
                [self._agent_cfg_to_img(self.agents[ag]) for ag in agent],0)
        elif agent == None:
            obs = torch.stack(
                [self._agent_cfg_to_img(ag) for ag in self.agents],0)
        else:
            raise ValueError

        return obs

    def _move_cardinal(self,agent, action):
        """
        Take actions wrt cardinal N-S-E-W directions.

        '0' : STOP
        '1' : E
        '2' : N
        '3' : W
        '4' : S
        """
        N = self.grid_size
        m = self.cell_occ_map

        r = agent.row
        c = agent.col
        tb = agent.tb

        if action == 0:
            # STAY
            pass

        elif action == 1:
            # East: +X Axis
            if (c + 1) < N and m[r][c+1] == 0:
                agent.col += 1

        elif action == 2:
            # North: -Y Axis
            if (r - 1) >= 0 and m[r-1][c] == 0:
               agent.row -= 1

        elif action == 3:
            # West: -X Axis
            if (c - 1) >= 0 and m[r][c-1] == 0:
                agent.col -= 1

        elif action == 4:
            # South: +Y Axis
            if (r + 1) < N and m[r+1][c] == 0:
                agent.row += 1

        else:
            raise ValueError('Invalid action passed')

    def _move_3(self,agent,action):
        """
        Take actions with triplet observations.
        """
        N = self.grid_size
        m = self.cell_occ_map

        r = agent.row
        c = agent.col
        tb = agent.tb

        if action == 3:
            # STAY
            pass

        elif action == 0:
            # Step Forward

            if tb == 0:
                # Agent is facing +Y Axis
                if (r + 1) < N and m[r+1][c] == 0:
                    agent.row += 1

            elif tb == 1:
                # Agent is facing -X Axis
                if (c - 1) >= 0 and m[r][c-1] == 0:
                    agent.col -= 1

            elif tb == 2:
                # Agent is facing -Y Axis
               if (r -1) >= 0 and m[r-1][c] == 0:
                   agent.row -= 1
            else:
                # Agent is facing +X Axis
                if (c + 1) < N and m[r][c+1] == 0:
                            agent.col += 1
        elif action == 1:
            # TURN BODY LEFT
            agent.tb = (agent.tb + 1) % self.num_body_turns

        elif action == 2:
            # TURN BODY RIGHT
            agent.tb = (agent.tb - 1) % self.num_body_turns

        else:
            raise ValueError('Invalid action passed')

    def _move_1(self,agent,action):
        """
        Take actions with single observations.
        """

        N = self.grid_size
        m = self.cell_occ_map

        r = agent.row
        c = agent.col
        th = agent.th
        tb = agent.tb

        if action == 5:
            # STAY
            pass

        elif action == 0:
            # STEP FORWARD

            if tb == 0:
                # Agent is facing +Y Axis
                if (r + 1) < N and m[r+1][c] == 0:
                    agent.row += 1

            elif tb == 1:
                # Agent is facing -X Axis
                if (c - 1) >= 0 and m[r][c-1] == 0:
                    agent.col -= 1

            elif tb == 2:
                # Agent is facing -Y Axis
               if (r -1) >= 0 and m[r-1][c] == 0:
                   agent.row -= 1
            else:
                # Agent is facing +X Axis
                if (c + 1) < N and m[r][c+1] == 0:
                    agent.col += 1

            # All the agent face straight after
            # taking a step
            agent.th = agent.tb

        elif action == 1:
            # TURN BODY LEFT
            agent.tb = (agent.tb + 1) % self.num_body_turns

        elif action == 2:
            # TURN BODY RIGHT
            agent.tb = (agent.tb - 1) % self.num_body_turns

        elif action == 3:
            # TURN HEAD LEFT
            agent.th = (agent.th + 1) % self.num_head_turns

        elif action == 4:
            # TURN HEAD RIGHT
            agent.th = (agent.th - 1) % self.num_head_turns

        else:
            raise ValueError('Invalid Actions')
