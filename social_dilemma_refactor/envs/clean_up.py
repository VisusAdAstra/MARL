import numpy as np
from gymnasium.spaces import Discrete
from numpy.random import rand

from envs.agent import CleanupAgent
from envs.map_env import MapEnv
from envs.maps import CLEANUP_MAP

# Add custom actions to the agents
# length of firing beam, length of cleanup beam
_CLEANUP_ACTIONS = {"FIRE": 5, "CLEAN": 5}
# Custom colour dictionary
CLEANUP_COLORS = {
    b"C": np.array([100, 255, 255], dtype=np.uint8),  # Cyan cleaning beam
    # Light grey-blue stream cell
    b"S": np.array([99, 156, 194], dtype=np.uint8),
    b"H": np.array([113, 75, 24], dtype=np.uint8),  # Brown waste cells
    # Light grey-blue river cell
    b"R": np.array([99, 156, 194], dtype=np.uint8),
}

CLEANUP_VIEW_SIZE = 7

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.125


class CleanupEnv(MapEnv):
    def __init__(
            self,
            ascii_map=CLEANUP_MAP,
            number_of_agents=5,
            number_of_inequity=0,
            return_agent_actions=False,
            use_collective_reward=False,
            inequity_averse_reward=False,
            alpha=0.0,
            beta=0.0,
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            number_of_agents,
            number_of_inequity,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )
        self.__action_space = self.get_action_space()
        self.action_spaces = {
            agent_id: self.__action_space for agent_id in self.agents}
        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])
        # print("..................")
        self.color_map.update(CLEANUP_COLORS)

        self.waste_density = 0
        self.spawn_apple_num = 0

    def get_action_space(self):
        return Discrete(9)

    def action_space(self, agent_id):
        return self.action_spaces[agent_id]

    # @property
    # def num_agents(self) -> int:
    #     assert len(self.agents) == self.num_of_agents
    #     return len(self.agents)

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        return updates

    def custom_map_update(self):
        """ "Update the probabilities and then spawn"""
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())

    def setup_agents(self):
        """Constructs all the agents in self.agents"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_of_agents):
            agent_id = "agent_" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agents = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id,
                spawn_point,
                rotation,
                map_with_agents,
                view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents_dict[agent_id] = agent
        self.agents = list(self.agents_dict.keys())
        self.possible_agents = self.agents

    def spawn_apples_and_waste(self):

        self.spawn_apple_num = 0
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

                    self.spawn_apple_num += 1

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            np.random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        self.waste_density = 0
        if self.potential_waste_area > 0:
            self.waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area

        if self.waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if self.waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (1 - (self.waste_density - thresholdRestoration) / (thresholdDepletion - thresholdRestoration)) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area

    def get_custom_infos(self):
        return {"waste_density": self.waste_density, "spawn_apple_num": self.spawn_apple_num}
