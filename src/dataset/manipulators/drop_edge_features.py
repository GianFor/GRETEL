import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class DropEdgeFeatures(BaseManipulator):
    """Replace the edge features with GraphInstance's default: one column of ones.

    The GCN oracles read only node features, structure and edge weights, so
    edge attributes such as RDKit bond types never reach the decision. Dropping
    them keeps the graph that is explained equal to the graph the oracle uses,
    and matches the counterfactuals that explainers build without edge features.
    """

    def process(self):
        self.dataset.edge_features_map = {}
        for instance in self.dataset.instances:
            self.process_instance(instance)

    def process_instance(self, instance):
        instance.edge_features = np.ones((np.count_nonzero(instance.data), 1), dtype=np.float32)
