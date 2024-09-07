import numpy as np
from typing import List, Dict, Tuple, Optional

class ProblemArea:
    """
    Represents a problem area with goals and conflicts between goals.

    Attributes:
        id (int): Unique identifier for the problem area.
        goals (np.ndarray): Array of goal IDs for this problem area.
        conflict_matrix (np.ndarray): Matrix of conflicts between goals.
    """

    def __init__(self, id: int, num_goals: int, conflict_matrix: Optional[np.ndarray] = None):
        """
        Initialize a ProblemArea.

        Args:
            id (int): Unique identifier for the problem area.
            num_goals (int): Number of goals for this problem area (not including the "no goal" goal).
            conflict_matrix (Optional[np.ndarray]): Matrix of conflicts between goals. If no matrix is provided, it will be initialized as a zero matrix. If a matrix is provided, it must be of size (num_goals, num_goals).
        """
        self.id = id
        self.goals = np.arange(num_goals + 1)  # Add 1 to include the "no goal" option
        if conflict_matrix is None:
            self.conflict_matrix = np.zeros((num_goals + 1, num_goals + 1))
        else:
            if conflict_matrix.shape[0] == num_goals and conflict_matrix.shape[1] == num_goals:
                # Add 1 to include the "no goal" option
                self.conflict_matrix = np.zeros((num_goals + 1, num_goals + 1))
                self.conflict_matrix[1:, 1:] = conflict_matrix
            elif conflict_matrix.shape[0] != num_goals + 1 or conflict_matrix.shape[1] != num_goals + 1:
                raise ValueError("Conflict matrix size must be (num_goals + 1, num_goals + 1) to account for each goal and the 'no goal' option.")
            else:
                self.conflict_matrix = conflict_matrix

    def __str__(self):
        """
        Return a string representation of the ProblemArea, showing its ID and goals.
        """
        return f"Problem Area {self.id}: Goals: {list(self.goals)} Conflict matrix: \n{self.conflict_matrix}"

class Agent:
    """
    Represents an agent with goals and weights for different problem areas.

    Attributes:
        id (int): Unique identifier for the agent.
        goals (np.ndarray): Array of goals for each problem area.
        weights (np.ndarray): Array of weights for each goal in each problem area.
    """

    def __init__(self, id: int, num_problem_areas: int):
        """
        Initialize an Agent.

        Args:
            id (int): Unique identifier for the agent.
            num_problem_areas (int): Number of problem areas in the world.
        """
        self.id = id
        self.goals = np.zeros(num_problem_areas, dtype=int)
        self.weights = np.zeros(num_problem_areas, dtype=float)

    def set_goal(self, pa_id: int, goal: int, weight: float):
        """
        Set a goal and weight for a specific problem area.

        Args:
            pa_id (int): ID of the problem area to set the goal for.
            goal (int): ID of the goal to set.
            weight (float): Weight to assign to the goal.
        """
        self.goals[pa_id] = goal
        self.weights[pa_id] = weight

    def normalize_weights(self):
        """Normalize the weights of all of this agent's goals so they sum to 1."""
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight

    def __str__(self):
        """
        Return a string representation of the Agent, showing its ID and goals for each problem area.
        """
        agent_str = f"Agent {self.id}:\n"
        for i in range(len(self.goals)):
            agent_str += f"  Problem Area {i}: Goal {self.goals[i]}, Weight {self.weight[i]:.2f}\n"
        return agent_str.strip()  # strip() removes the trailing newline

class Group:
    """
    Represents a group of agents with the same goal and weight category for a specific problem area.

    Attributes:
        pa_id (int): ID of the problem area this group is associated with.
        goal_id (int): ID of the goal for this group.
        weight_category (int): Weight category of this group.
        agent_ids (set): Set of agent IDs in this group.
    """

    def __init__(self, pa_id: int, goal_id: int, weight_category: int = 0):
        self.pa_id = pa_id
        self.goal_id = goal_id
        self.weight_category = weight_category
        self.agent_ids = set()

    def add_agent(self, agent_id: int):
        """Add an agent to the group."""
        self.agent_ids.add(agent_id)

    def remove_agent(self, agent_id: int):
        """Remove an agent from the group."""
        self.agent_ids.discard(agent_id)

    def decrement_agent_ids(self, from_id: int):
        """Decrement the agent IDs in the group that are greater than or equal to a given ID."""
        new_ids = {a_id - 1 if a_id >= from_id else a_id for a_id in self.agent_ids}
        self.agent_ids = new_ids

    def size(self) -> int:
        """Return the number of agents in the group."""
        return len(self.agent_ids)

    def __repr__(self):
        return f"Group(PA:{self.pa_id}, Goal:{self.goal_id}, WeightCat:{self.weight_category}, Agents:{len(self.agent_ids)})"

class World:
    """
    Represents the entire world, containing problem areas and agents.

    This class manages the creation, modification, and analysis of problem areas and agents.

    Attributes:
        problem_areas (Dict[int, ProblemArea]): Dictionary of problem areas, keyed by their IDs.
        agents (List[Agent]): List of agents in the world.
        groups (Dict[int, Dict[Tuple[int, int], Group]]): Nested dictionary of groups.
        next_pa_id (int): Counter for generating unique problem area IDs.
        round_robin_trackers (Dict[int, int]): Dictionary of round-robin trackers for each problem area.
        num_problem_areas (int): Number of problem areas in the world.
    """

    def __init__(self):
        self.problem_areas: Dict[int, ProblemArea] = {}
        self.agents: List[Agent] = []
        self.groups: Dict[int, Dict[Tuple[int, int], Group]] = {} # TODO: Make this data structure more efficient
        self.next_pa_id = 0
        self.round_robin_trackers: Dict[int, int] = {pa_id: 1 for pa_id in self.problem_areas.keys()}
        self.num_problem_areas = 0
        self.misalignment_cache = {}

    def add_problem_area(self, num_goals: int = 2,
                         conflict_matrix: Optional[np.ndarray] = None,
                         single_conflict_value: Optional[float] = None,
                         randomize_conflict: bool = False,
                         random_conflict_range: Tuple[float, float] = (0.0,1)) -> int:
        """
        Add a new problem area to the world. The conflict matrix can be provided, randomly generated, or initialized with zeroes.

        Args:
            num_goals [int]: Number of goals for this problem area (not including the "no goal" goal).
            conflict_matrix (Optional[np.ndarray]): Pre-defined conflict matrix of size (num_goals, num_goals).
            single_conflict_value (Optional[float]): Single conflict value to use for all goal pairs. If provided, this will create or override the conflict matrix.
            randomize_conflict (bool): Whether to randomly generate conflict values for this problem area.
            random_conflict_range (Tuple[float, float]): Range of conflict values (min, max). Must be in range [0, 1].

        Returns:
            int: ID of the newly created problem area.
        """
        if randomize_conflict:
            if conflict_matrix is not None:
                raise ValueError("conflict_matrix must be None if randomize_conflict is True.")
            # Randomly generate a symmetric conflict matrix
            conflict_matrix = np.random.uniform(*random_conflict_range, size=(num_goals, num_goals))
            # Just use the upper triangle and mirror it to the lower triangle
            conflict_matrix = np.triu(conflict_matrix) + np.triu(conflict_matrix, 1).T
            # Set the diagonal to 0
            np.fill_diagonal(conflict_matrix, 0)
        elif single_conflict_value is not None:
            conflict_matrix = np.full((num_goals, num_goals), single_conflict_value)
            np.fill_diagonal(conflict_matrix, 0)
        elif conflict_matrix is None:
            raise ValueError("conflict_matrix must be provided if randomize_conflict is False and single_conflict_value is None.")
        elif conflict_matrix.shape[0] != num_goals or conflict_matrix.shape[1] != num_goals:
            raise ValueError("Conflict matrix size must be (num_goals, num_goals) to account for each goal.")
        
        pa_id = self.next_pa_id
        self.next_pa_id += 1
        new_pa = ProblemArea(pa_id, num_goals, conflict_matrix)
        self.problem_areas[pa_id] = new_pa
        self.num_problem_areas += 1

        # Update all existing agents with a new slot for this problem area
        for agent in self.agents:
            agent.goals = np.pad(agent.goals, (0, 1), 'constant')
            agent.weights = np.pad(agent.weights, (0, 1), 'constant') 

        return pa_id
    
    def assign_specific_goal_for_problem_area(self, pa_id: int,
                                               goal_id: int = 0,
                                               weight: float = 0,
                                               agent_ids: Optional[List[int]]=None,
                                               update_groups: bool = False,
                                               normalize_weights: bool = False):
        """
        Assign a specific goal to all (or a list of) existing agents for a specific problem area. By default, assigns the "no goal" option with 0 weight.

        Args:
            pa_id (int): ID of the problem area to assign goals for.
            goal_id (int): ID of the goal to assign.
            weight (float): Weight to assign to the goal.
            agent_ids (Optional[List[int]]): List of agent IDs to assign goals to. If None, assign goals to all agents.
            update_groups (bool): Whether to update groups after assigning goals.
            normalize_weights (bool): Whether to normalize the weights of each agent's goals.
        """
        if pa_id not in self.problem_areas:
            raise ValueError(f"Problem area with ID {pa_id} not found.")
        if agent_ids is None:
            agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            self.update_agent(agent_id, {pa_id: (goal_id, weight)}, update_groups, normalize_weights)

    def assign_goals_round_robin_in_problem_area(self, pa_id: int,
                                                 agent_ids: Optional[List[int]] = None,
                                                 start_goal: Optional[int] = None,
                                                 weights_for_goals: Optional[Dict[int, float]] = None,
                                                 update_groups: bool = False,
                                                 allow_no_goal: bool = False,
                                                 normalize_weights: bool = False) -> int:
        """
        Assign goals to all (or a list of) existing agents for a specific problem area in a round-robin fashion.

        Args:
            pa_id (int): ID of the problem area to assign goals for.
            agent_ids (Optional[List[int]]): List of agent IDs to assign goals to. If None, assign goals to all agents.
            start_goal (Optional[int]): ID of the goal to start assigning from. If None, start from the global round-robin tracker for this problem area.
            weights_for_goals (Dict[int: float]): Dictionary mapping goal IDs to the weights they should be assigned with. If None, assign a weight of 1 to all goals (except the "no goal" option, which is assigned a weight of 0).
            update_groups (bool): Whether to update groups after assigning goals.
            allow_no_goal (bool): Whether to allow agents to be assigned the "no goal" option for this problem area.
            normalize_weights (bool): Whether to normalize the weights of each agent's goals.

        Returns:
            int: ID of the goal to start assigning from next time, which is also stored in the global round-robin tracker for this problem area.
        """
        if pa_id not in self.problem_areas:
            raise ValueError(f"Problem area with ID {pa_id} not found.")
        if agent_ids is None:
            agent_ids = range(len(self.agents)) # Assign goals to all agents if no list is provided

        goal_ids = self.problem_areas[pa_id].goals
        if weights_for_goals is None:
            # If no weights are provided, assign equal weights to all goals (except the "no goal" option)
            weights_for_goals = {goal: 1 for goal in goal_ids[1:]}
            weights_for_goals[0] = 0
        else:
            # If some weights are provided, but not all, fill in the missing ones with 1 (and a 0 for the "no goal" option)
            if 0 not in weights_for_goals:
                weights_for_goals[0] = 0
            for goal in goal_ids:
                if goal not in weights_for_goals:
                    weights_for_goals[goal] = 1

        if start_goal is None:
            if pa_id not in self.round_robin_trackers:
                start_goal = 1
            else:        
                start_goal = self.round_robin_trackers[pa_id]

        for i, agent_id in enumerate(agent_ids):
            if allow_no_goal:
                goal_id = goal_ids[(start_goal + i) % len(goal_ids)]
            else:
                goal_id = goal_ids[(start_goal + i) % (len(goal_ids) - 1) + 1]
            weight = weights_for_goals[goal_id]
            self.update_agent(agent_id, {pa_id: (goal_id, weight)}, update_groups, normalize_weights)

        self.round_robin_trackers[pa_id] = (start_goal + len(agent_ids)) % (len(goal_ids)-1) + 1
        return self.round_robin_trackers[pa_id]

    def assign_random_goals_for_problem_area(self, pa_id: int,
                                             agent_ids: Optional[List[int]]=None,
                                             weight_range: Tuple[float, float]=(0,1),
                                             allow_no_goal: bool = False,
                                             update_groups: bool = False,
                                             normalize_weights: bool = False):
        """
        Assign random goals to all (or a list of) existing agents for a specific problem area.

        Args:
            pa_id (int): ID of the problem area to assign goals for.
            agent_ids (Optional[List[int]]): List of agent IDs to assign goals to. If None, assign goals to all agents.
            weight_range (Tuple[float, float]): Range of weight values (min, max).
            allow_no_goal (bool): Whether to allow agents to have no goal for this problem area.
            update_groups (bool): Whether to update groups after assigning goals.
            normalize_weights (bool): Whether to normalize the weights of each agent's goals.
        """
        if pa_id not in self.problem_areas:
            raise ValueError(f"Problem area with ID {pa_id} not found.")
        if agent_ids is None:
            agent_ids = list(self.agents.keys())
        goal_ids = self.problem_areas[pa_id].goals
        for agent_id in agent_ids:
            if allow_no_goal:
                goal = np.random.choice(goal_ids)
            else:
                goal = np.random.choice(goal_ids[1:])
            weight = np.random.uniform(*weight_range)
            self.update_agent(agent_id, {pa_id: (goal, weight)}, normalize_weights=normalize_weights, update_groups=update_groups)

    def remove_problem_area(self, pa_id: int):
        """
        Remove a problem area from the world and update all agents accordingly.

        Args:
            pa_id (int): ID of the problem area to remove.
        """
        if pa_id in self.problem_areas:
            del self.problem_areas[pa_id]
            for agent in self.agents.values():
                agent.goals.pop(pa_id, None)
            self.groups.pop(pa_id, None)

    def add_agents_in_bulk(self, num_agents: int,
                           specified_goals: Optional[Dict[int, Tuple[int, float]]] = None,
                           random_goals: bool = False,
                           allow_no_goal: bool = False,
                           random_weight_range: Tuple[float, float] = (1, 1),
                           round_robin_goals: bool = False,
                           normalize_weights: bool = False):
        """
        Add multiple agents to the world, with a random or specified goal for each PA, and a random or specified weight.

        Args:
            num_agents (int): Number of agents to add.
            specified_goals (Optional[Dict[int, Tuple[int, float]]]): Pre-defined goals for the agents. Format: {pa_id: (goal_id, weight)}.
            random_goals (bool): Whether to randomly assign goals if no specified goals are provided. If false, agents will be assigned the "no goal" option.
            random_allow_no_goal (bool): Whether to allow agents to be randomly assigned the "no goal" option for problem areas. Does nothing if random_goals is False.
            random_weight_range (Tuple[float, float]): Range of weight values (min, max) to use if randomly assigning goals. Always assigns a weight of 1 if unspecified.
            round_robin_goals (bool): Whether to assign goals in a round-robin fashion for each problem area.
        """
        if specified_goals is not None:
            for _ in range(num_agents):
                new_agent_id = self.add_agent(specified_goals, normalize_weights=False, add_to_group=False)
                if normalize_weights:
                    self.agents[new_agent_id].normalize_weights()
        else:
            # Add all of the agents, then assign goals and weights by problem area
            new_agent_ids = []
            for _ in range(num_agents):
                new_agent_ids.append(self.add_agent({}))
            for pa_id in self.problem_areas:
                if round_robin_goals:
                    self.assign_goals_round_robin_in_problem_area(pa_id, new_agent_ids, allow_no_goal=allow_no_goal, update_groups=False, normalize_weights=normalize_weights)
                elif random_goals:
                    self.assign_random_goals_for_problem_area(pa_id, new_agent_ids, weight_range=random_weight_range, allow_no_goal=allow_no_goal, update_groups=False, normalize_weights=normalize_weights)
                else:
                    self.assign_specific_goal_for_problem_area(pa_id, goal_id=0, weight=0, agent_ids=new_agent_ids, update_groups=False, normalize_weights=normalize_weights)


    def add_agent(self, goals: Dict[int, Tuple[int, float]], normalize_weights: bool = False, add_to_group: bool = True) -> int:
        """
        Add a single agent to the world and update groups.

        Args:
            goals (Dict[int, Tuple[int, float]]): Goals for the new agent.
            normalize_weights (bool): Whether to normalize the weights of the agent's goals.
            add_to_group (bool): Whether to add the agent to the appropriate group.

        Returns:
            int: ID of the newly created agent.
        """
        agent_id = len(self.agents)
        new_agent = Agent(agent_id, self.num_problem_areas)
        for pa_id, (goal_id, weight) in goals.items():
            new_agent.set_goal(pa_id, goal_id, weight)

        if normalize_weights:
            new_agent.normalize_weights()
        
        if add_to_group:
            for pa_id, (goal_id, _) in goals.items():
                self._add_agent_to_group(agent_id, pa_id, goal_id)
        
        self.agents.append(new_agent)

        return agent_id

    def update_agent(self, agent_id: int, new_goals: Dict[int, Tuple[int, float]], normalize_weights: bool = False, update_groups: bool = True):
        """
        Update the goals of an existing agent.

        Args:
            agent_id (int): ID of the agent to update.
            new_goals (Dict[int, Tuple[int, float]]): New goals for the agent.
            normalize_weights (bool): Whether to normalize the weights of the agent's goals.
            update_groups (bool): Whether to remove the agent from old groups and add it to its new groups.
        """
        if 0 <= agent_id < len(self.agents):
            agent = self.agents[agent_id]
            
            if update_groups:
                for pa_id, goal in enumerate(agent.goals):
                    self._remove_agent_from_group(agent_id, pa_id, goal)

            for pa_id, (goal, weight) in new_goals.items():
                if 0 <= pa_id < self.num_problem_areas:
                    agent.goals[pa_id] = goal
                    agent.weights[pa_id] = weight
                else:
                    raise ValueError(f"Invalid problem area ID: {pa_id}")

            if normalize_weights:
                agent.normalize_weights()

            if update_groups:
                for pa_id, goal in enumerate(agent.goals):
                    self._add_agent_to_group(agent_id, pa_id, goal)
                
            # Clear the misalignment cache for this agent
            self.misalignment_cache = {k: v for k, v in self.misalignment_cache.items() 
                                       if agent_id not in k[:2]} # Remove all comparisons involving this agent

        else:
            raise ValueError(f"Agent with ID {agent_id} not found.")

    def remove_agent(self, agent_id: int, update_groups: bool = True, update_cache: bool = True):
        """
        Remove an agent from the world and update groups.

        Args:
            agent_id (int): ID of the agent to remove.
            update_groups (bool): Whether to update groups after removing the agent.
            update_cache (bool): Whether to update the misalignment cache to account for the new agent IDs. Clears the cache of affected agents otherwise.
        """
        if 0 <= agent_id < len(self.agents):
            # Remove the agent from all groups
            for i in range(self.num_problem_areas):
                self._remove_agent_from_group(agent_id, i, self.agents[agent_id].goals[i])

            # Remove the agent
            self.agents.pop(agent_id)

            # Update the IDs of the remaining agents
            for i, agent in enumerate(self.agents[agent_id:], start=agent_id):
                agent.id = i

            
            if update_groups:
                # Update all group memberships for all agents with IDs >= agent_id
                for pa_id in self.groups:
                    for goal_id in list(self.groups[pa_id].keys()):
                        self.groups[pa_id][goal_id].decrement_agent_ids(agent_id)

            if update_cache:
                new_cache = {}
                for (a1, a2, pa_id), value in self.misalignment_cache.items():
                    if a1 == agent_id or a2 == agent_id:
                        continue # Skip comparisons involving the removed agent in order to remove them from the cache
                    elif a1 >= agent_id and a2 >= agent_id:
                        new_cache[(a1-1, a2-1, pa_id)] = value
                    elif a1 >= agent_id:
                        new_cache[(a1-1, a2, pa_id)] = value
                    elif a2 >= agent_id:
                        new_cache[(a1, a2-1, pa_id)] = value
                    elif a1 != agent_id and a2 != agent_id:
                        new_cache[(a1, a2, pa_id)] = value
                self.misalignment_cache = new_cache
            else:
                # Clear the misalignment cache of agents that have new IDs
                for a_id in range(agent_id, len(self.agents)):
                    self.misalignment_cache = {k: v for k, v in self.misalignment_cache.items()
                                                 if a_id not in k[:2]}

        else:
            raise ValueError(f"Agent with ID {agent_id} not found.")

    def remove_agents_in_bulk(self, agent_ids: List[int], update_groups: bool = True):
        """
        Remove multiple agents from the world and update groups.

        Args:
            agent_ids (List[int]): List of agent IDs to remove.
            update_groups (bool): Whether to update groups after removing the agents.
        """
        for agent_id in sorted(agent_ids, reverse=True):
            self.remove_agent(agent_id, update_groups=False, update_cache=False)
        if update_groups:
            self._update_group_memberships(0)

    def normalize_all_agent_weights(self):
        """Normalize weights for all agents in the world."""
        for agent in self.agents.values():
            agent.normalize_weights()

    def _add_agent_to_group(self, agent_id: int, pa_id: int, goal_id: int):
        """Helper method to add an agent to the appropriate group.
        TODO: Account for weight categories. Weight categories are currently not used in the paper.
        """
        if pa_id not in self.groups:
            self.groups[pa_id] = {}
        
        key = (goal_id, 0)  # Default to weight category 0 (no thresholds)
        if key not in self.groups[pa_id]:
            self.groups[pa_id][key] = Group(pa_id, goal_id)
        
        self.groups[pa_id][key].add_agent(agent_id)

    def _remove_agent_from_group(self, agent_id: int, pa_id: int, goal_id: int):
        """Helper method to remove an agent from its group.
        TODO: Account for weight categories. Weight categories are currently not used in the paper.
        """
        if pa_id in self.groups:
            key = (goal_id, 0)  # Default to weight category 0 (no thresholds)
            if key in self.groups[pa_id]:
                self.groups[pa_id][key].remove_agent(agent_id)
                if self.groups[pa_id][key].size() == 0:
                    del self.groups[pa_id][key]
            if not self.groups[pa_id]:
                del self.groups[pa_id]

    def update_conflict(self, pa_id: int, goal1: int, goal2: int, new_conflict: float):
        """
        Update the conflict between two goals in a problem area.

        Args:
            pa_id (int): ID of the problem area.
            goal1 (int): ID of the first goal.
            goal2 (int): ID of the second goal.
            new_conflict (float): New conflict value between the goals.
        """
        if pa_id in self.problem_areas:
            pa = self.problem_areas[pa_id]
            pa.conflict_matrix[goal1, goal2] = pa.conflict_matrix[goal2, goal1] = new_conflict

    def update_all_groups(self, weight_thresholds: Optional[List[float]] = None):
        """
        Update all groups in the world, optionally considering weight thresholds.

        Args:
            weight_thresholds (Optional[List[float]]): List of weight thresholds for splitting groups.
        """
        for pa_id in self.problem_areas:
            self.groups[pa_id] = self.update_pa_groups(pa_id, weight_thresholds)

    def update_pa_groups(self, pa_id: int, weight_thresholds: Optional[List[float]] = [0]):
        """
        Create or update groups for a specific problem area, optionally considering weight thresholds.

        Args:
            pa_id (int): ID of the problem area to update groups for.
            weight_thresholds (Optional[List[float]]): List of weight thresholds for splitting groups. If [0], every agent with the same goal is in the same group.
        Raises:
            ValueError: If the problem area ID is not found or if weight thresholds are invalid.
        """
        if pa_id not in self.problem_areas:
            raise ValueError(f"Problem area with ID {pa_id} not found.")

        if not weight_thresholds:
            weight_thresholds = [0]

        if not all(0 < t < 1 for t in weight_thresholds):
            if weight_thresholds == [0]:
                pass # Allow a single threshold of 0
            else:
                raise ValueError("All weight thresholds must be between 0 and 1.")
        
        weight_thresholds.sort()
        new_groups = {}
        for agent_id, agent in self.agents.items():
            if pa_id in agent.goals:
                goal_id, weight = agent.goals[pa_id]
                weight_category = sum(weight >= t for t in weight_thresholds)
                key = (goal_id, weight_category)
                if key not in new_groups:
                    new_groups[key] = Group(pa_id, goal_id, weight_category)
                new_groups[key].add_agent(agent_id)
        self.groups[pa_id] = new_groups

    def get_average_pa_weight(self, pa_id: int) -> float:
        """
        Get the average weight of all agents in a specific problem area.

        Args:
            pa_id (int): ID of the problem area to calculate the average weight for.

        Returns:
            float: Average weight of agents in the problem area.
        """
        if pa_id not in self.problem_areas:
            raise ValueError(f"Problem area with ID {pa_id} not found.")
        return np.mean([agent.weights[pa_id] for agent in self.agents])

    def calculate_population_misalignment(self, problem_area_id: Optional[int] = None, 
                                          normalize_by_goal_number: bool = False, 
                                          weight_averaging_style: str = "geometric mean", 
                                          pa_misalignment_averaging_style: str = "arithmetic mean") -> float:
        """
        Calculate the overall misalignment of the entire population of agents.

        Args:
            problem_area_id (Optional[int]): ID of the problem area to consider (if None, consider all).
            normalize_by_goal_number (bool): Whether to normalize the misalignment by the number of goals in the problem area.
            weight_averaging_style (str): Style of averaging to use for weights. Choose one of "arithmetic mean", "geometric mean", "harmonic mean", "min", "max", or "disregard weights".
            pa_misalignment_averaging_style (str): Style of averaging to use for weights across all problem areas. Choose one of "arithmetic mean", "geometric mean", "harmonic mean", "min", "max".

        Returns:
            float: Overall population misalignment value (0 to 1).
        """
        n = len(self.agents)
        if n <= 1:
            return 0.0

        if problem_area_id is None:
            # Calculate misalignment across all problem areas
            misalignments = []
            for pa_id in self.problem_areas.keys():
                pa_misalignment = self.calculate_population_misalignment(pa_id, normalize_by_goal_number, weight_averaging_style)
                misalignments.append(pa_misalignment)

            averaged_misalignment = None
            if pa_misalignment_averaging_style == "arithmetic mean":
                averaged_misalignment = np.mean(misalignments)
            elif pa_misalignment_averaging_style == "geometric mean":
                averaged_misalignment = np.prod(misalignments) ** (1 / len(misalignments))
            elif pa_misalignment_averaging_style == "harmonic mean":
                averaged_misalignment = 1 / np.mean(1 / np.array(misalignments))
            elif pa_misalignment_averaging_style == "min":
                averaged_misalignment = np.min(misalignments)
            elif pa_misalignment_averaging_style == "max":
                averaged_misalignment = np.max(misalignments)
            return averaged_misalignment
        else:
            # Calculate misalignment for a specific problem area
            if problem_area_id < 0 or problem_area_id >= self.num_problem_areas:
                raise ValueError(f"Invalid problem area ID: {problem_area_id}")

            goals = np.array([agent.goals[problem_area_id] for agent in self.agents])
            weights = np.array([agent.weights[problem_area_id] for agent in self.agents])

            # Create a 2D array of conflicts between each pair of agents in this problem area
            conflicts = self.problem_areas[problem_area_id].conflict_matrix[goals][:, goals]

            # Calculate misalignment between each pair of agents by scaling each conflict by the average weight of the two agents
            misalignment_matrix = conflicts 

            if weight_averaging_style == "arithmetic mean":
                misalignment_matrix = misalignment_matrix * (weights[:, np.newaxis] + weights) / 2
            elif weight_averaging_style == "geometric mean":
                misalignment_matrix = misalignment_matrix * np.sqrt(weights[:, np.newaxis] * weights) # In Latex: \sqrt{w_i w_j}
            elif weight_averaging_style == "harmonic mean":
                misalignment_matrix = misalignment_matrix * 2 / (1 / weights[:, np.newaxis] + 1 / weights)
            elif weight_averaging_style == "min":
                misalignment_matrix = misalignment_matrix * np.minimum(weights[:, np.newaxis], weights)
            elif weight_averaging_style == "max":
                misalignment_matrix = misalignment_matrix * np.maximum(weights[:, np.newaxis], weights)
            elif weight_averaging_style == "disregard weights":
                pass # Just use the conflict values. Equivalent to setting all weights to 1.
            else:
                raise ValueError("Invalid weight averaging style. Choose one of 'arithmetic mean', 'geometric mean', 'harmonic mean', 'min', 'max', or 'disregard weights'.")

            # Sum up the upper triangle of the misalignment matrix
            total_misalignment = np.sum(np.triu(misalignment_matrix, k=1)) # This is the sum of the misalignment between each unique pair of agents in this problem area
            sampled_misalignment = total_misalignment / (n * (n - 1) / 2) # Normalize by the number of unique pairs of agents

            if normalize_by_goal_number:
                # Normalize by the number of goals in this problem area
                k = len(self.problem_areas[problem_area_id].goals) - 1 # Subtract 1 to exclude the "no goal" option
                return sampled_misalignment * k / (k - 1)
            else:
                return sampled_misalignment
