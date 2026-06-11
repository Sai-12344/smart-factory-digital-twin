import numpy as np
import networkx as nx


class FactoryGraph:
    """
    Builds a factory machine dependency graph.
    Each machine is a node.
    Edges represent:
    - load dependency
    - process sequence
    - shared components
    - layout proximity
    """

    def __init__(self, num_machines=10):
        self.num_machines = num_machines
        self.graph = nx.Graph()

        # Initialize graph
        for i in range(num_machines):
            self.graph.add_node(i)

        self._create_random_edges()

    def _create_random_edges(self):
        """
        Randomly generate machine interactions.
        Higher-level version: Real factories use layout + process sequence.
        """
        for i in range(self.num_machines):
            for j in range(i + 1, self.num_machines):
                # Random chance of connection
                if np.random.rand() < 0.25:
                    self.graph.add_edge(i, j, weight=np.random.uniform(0.3, 1.0))

    def adjacency_matrix(self):
        """Returns adjacency matrix for GNN."""
        return nx.to_numpy_array(self.graph)

    def node_features(self, machine_states):
        """
        machine_states: list of machine dicts from digital twin
        Extracts node features for GNN:
        - temp
        - vibration
        - load
        - wear
        - failure_prob
        """

        features = []

        for machine in machine_states:
            features.append([
                machine["temp"],
                machine["vibration"],
                machine["load"],
                machine["wear"],
                machine["failure_prob"],
            ])

        return np.array(features)

    def to_graph_data(self, machine_states):
        """
        Output for GNN training/inference:
        - adjacency matrix
        - node feature matrix
        """
        A = self.adjacency_matrix()
        X = self.node_features(machine_states)
        return A, X