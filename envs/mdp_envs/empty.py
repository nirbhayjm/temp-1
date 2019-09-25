from gym_minigrid.minigrid import *
from ..minigrid_nav import *

class EmptyGridNav(NavGridEnv):
    """
    Simple grid world for point navigation
    """

    class CardinalActions(IntEnum):

        right = 1
        down = 2
        left = 3
        up = 4

        # Done completing task
        done = 5

    def __init__(self, size=5, max_steps=10):
        super().__init__(
            grid_size=size,
            max_steps=max_steps,
        )
        self.actions = EmptyGridWordl
        self.action_space = spaces.Discrete(len(self.actions))

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

        self.mission = "get to the green goal square"
