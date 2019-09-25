from gym_minigrid.minigrid import *
from ..minigrid_nav import *
import numpy as np

WALL_TYPE = np.int8
WALL = 0
EMPTY = 1
RED = 2
BLUE = 3

class MazeBase:
    def __init__(self, rows, columns):
        assert rows >= 1 and columns >= 1

        self.nrows = rows
        self.ncolumns = columns
        self.board = np.zeros((rows, columns), dtype=WALL_TYPE)
        self.board.fill(EMPTY)

    def set_borders(self):
        self.board[0, :] = self.board[-1, :] = WALL
        self.board[:, 0] = self.board[:, -1] = WALL

    def is_wall(self, x, y):
        assert self.in_maze(x, y)
        return self.board[x][y] == WALL

    def set_wall(self, x, y):
        assert self.in_maze(x, y)
        self.board[x][y] = WALL

    def remove_wall(self, x, y):
        assert self.in_maze(x, y)
        self.board[x][y] = EMPTY

    def in_maze(self, x, y):
        return 0 <= x < self.nrows and 0 <= y < self.ncolumns

    @staticmethod
    def create_maze(rows, columns, rng, complexity=0.9, density=.9):
        rows = (rows // 2) * 2 + 1
        columns = (columns // 2) * 2 + 1

        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (rows + columns)))
        density = int(density * ((rows // 2) * (columns // 2)))

        maze = MazeBase(rows, columns)
        maze.set_borders()

        # Make aisles
        for i in range(density):
            x, y = rng.randint(0, rows // 2) * 2, rng.randint(0, columns // 2) * 2
            maze.set_wall(x, y)

            for j in range(complexity):
                neighbours = []

                if maze.in_maze(x - 2, y):
                    neighbours.append((x - 2, y))

                if maze.in_maze(x + 2, y):
                    neighbours.append((x + 2, y))

                if maze.in_maze(x, y - 2):
                    neighbours.append((x, y - 2))

                if maze.in_maze(x, y + 2):
                    neighbours.append((x, y + 2))

                if len(neighbours):
                    next_x, next_y = neighbours[rng.random_integers(0, len(neighbours) - 1)]

                    if not maze.is_wall(next_x, next_y):
                        maze.set_wall(next_x, next_y)
                        maze.set_wall(next_x + (x - next_x) // 2, next_y + (y - next_y) // 2)
                        x, y = next_x, next_y

        return maze

class Maze(NavGridEnv):
    """
    Randomly generated grid
    """

    def __init__(
        self,
        size,
        complexity=0.9,
        density=0.9,
        max_steps=100,
        use_grid_in_state=False,
        normalize_agent_pos=False,
        obs_alpha=0.001,
        reward_scale=1.0,
        spawn='random',
        seed=123,
        reset_prob=1.0,
        render_rgb = False,
        term_prob=0.0,
        perturb_prob=0.0,
        static_grid=True,
        config_seed=13,
    ):
        self.density = density
        self.complexity = complexity
        self.spawn = spawn
        # self.config_seed = config_seed
        # self.maze_rng = np.random.RandomState()
        # self.maze_rng.seed(self.config_seed)
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
            term_prob=term_prob,
            render_rgb=render_rgb,
            static_grid=static_grid,
            config_seed=config_seed,
            seed=seed,
            perturb_prob=perturb_prob,
            )

        new_spaces = self.observation_space.spaces
        new_spaces.update({
            'mission': spaces.MultiDiscrete([size, size]),
        })
        self.observation_space = spaces.Dict(new_spaces)

    def _gen_grid(self, width, height):
        maze = MazeBase.create_maze(
            height,
            width,
            rng=self.config_rng,
            complexity=self.complexity,
            density=self.density,
        )

        height, width = maze.board.shape
        # Create empty grid.
        self.grid = Grid(width, height)

        # Draw rectangular wall around the grid
        #self.grid.wall_rect(0, 0, width, height)

        for i in range(height):
            for j in range(width):
                if(maze.board[i][j] == 0):
                    self.grid.set(j, height - 1 - i, Wall())

        empty_y, empty_x = np.where(maze.board == 1)
        empty = [tup for tup in zip(empty_x, empty_y)]

        start_x, start_y = empty[self.config_rng.choice(len(empty))]

        # Because of difference in (i, j) and (x, y) indexing
        start_y = height - 1 - start_y
        self.start_pos = (start_x, start_y)
        self.start_dir = 0
        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir

        center = (self.grid.height)//2
        object_pos = [
            # [(1+width)//2, center],
            [width - 1, center - 1],
            [width - 1, center + 1],
        ]

        self.goal_index = int(self._rand_bool())
        self.mission = np.array(object_pos[self.goal_index],
            dtype='int64')

    def reset_agent_pos(self):
        if self.spawn == 'random':
            self.place_agent()
        self.agent_pos = self.start_pos
        self.start_dir = 0
        self.agent_dir = self.start_dir
