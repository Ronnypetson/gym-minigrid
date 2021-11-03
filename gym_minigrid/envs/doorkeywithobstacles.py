from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from operator import add


class DoorKeyObstEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward, with 0 or n obstacles
    """

    def __init__(self, size=7, n_obstacles=1, key_pos=(1, 1)):

        # Reduce obstacles if there are too many
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)

        self._key_default_pos = np.array(key_pos)

        super().__init__(
            grid_size=size,
            max_steps=5 * size * size
        )

        # Only 5 actions permitted: left, right, forward, pickup, tooggle
        self.action_space = spaces.Discrete(self.actions.drop + 1)

        self.reward_range = (-1, 1)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = math.floor(width / 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place a door in the wall
        doorIdx = 1
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.put_obj(Key('yellow'), *self._key_default_pos)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place obstacles
        # on the right side of the splitting wall
        self.obstacles = []
        top = (splitIdx + 1, 1)
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], size=(splitIdx, height), max_tries=100)

        self.mission = "use the key to open the door and then get to the goal, avoid obstacles"

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # drop is not used, it is mapped to toggle instead
        # map drop action to toggle
        if action == self.actions.drop:
            action = self.actions.toggle

        # Check if there is a ball in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type == 'ball'

        # If the agent tried to walk over an obstacle
        if action == self.actions.forward and not_clear:
            reward = -1
            done = True
            obs = self.gen_obs()
            info = {}
            return obs, reward, done, info

        # Update the agent's position/direction
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))
            top = (max(-1, top[0]), max(-1, top[1]))

            try:
                self.place_obj(self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100)
                self.grid.set(*old_pos, None)
            except:
                pass

        # generate observation after obstacle positions are updated
        obs = self.gen_obs()

        return obs, reward, done, info


# register classes of stochastic environments with obstacles

class DoorKeyObstEnv6x6(DoorKeyObstEnv):
    def __init__(self):
        super().__init__(size=6, n_obstacles=1)


class DoorKeyObstEnv8x8(DoorKeyObstEnv):
    def __init__(self):
        super().__init__(size=8, n_obstacles=1)


class DoorKeyObstEnv17x17(DoorKeyObstEnv):
    def __init__(self):
        super().__init__(size=17, n_obstacles=3)


# register classes of deterministic environments without obstacles

class DoorKeyNoObstEnv6x6(DoorKeyObstEnv):
    def __init__(self):
        super().__init__(size=6, n_obstacles=0)


class DoorKeyNoObstEnv7x7(DoorKeyObstEnv):
    def __init__(self):
        super().__init__(size=7, n_obstacles=0)


class DoorKeyNoObstEnv8x8(DoorKeyObstEnv):
    def __init__(self):
        super().__init__(size=8, n_obstacles=0)


class DoorKeyNoObstEnv17x17(DoorKeyObstEnv):
    def __init__(self):
        super().__init__(size=17, n_obstacles=0)


# register stochastic environments with obstacles


register(
    id='MiniGrid-DoorKeyObst-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyObstEnv6x6'
)

register(
    id='MiniGrid-DoorKeyObst-7x7-v0',
    entry_point='gym_minigrid.envs:DoorKeyObstEnv'
)

register(
    id='MiniGrid-DoorKeyObst-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyObstEnv8x8'
)

register(
    id='MiniGrid-DoorKeyObst-17x17-v0',
    entry_point='gym_minigrid.envs:DoorKeyObstEnv17x17'
)

# register deterministic environments without obstacles

register(
    id='MiniGrid-DoorKeyNoObst-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyNoObstEnv6x6'
)

register(
    id='MiniGrid-DoorKeyNoObst-7x7-v0',
    entry_point='gym_minigrid.envs:DoorKeyNoObstEnv7x7'
)

register(
    id='MiniGrid-DoorKeyNoObst-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyNoObstEnv8x8'
)

register(
    id='MiniGrid-DoorKeyNoObst-17x17-v0',
    entry_point='gym_minigrid.envs:DoorKeyNoObstEnv17x17'
)
