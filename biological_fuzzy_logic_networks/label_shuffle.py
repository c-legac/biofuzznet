import random
from collections.abc import Sequence
from typing import Type

from biological_fuzzy_logic_networks.biofuzznet import BioFuzzNet


class LabelShuffleMixin:
    """A BFN mixin that remaps biological nodes in the network (only those with a ground truth)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_mapping = None

    def setup_node_mapping(self, node_names: Sequence[str]):
        shuffled_names = random.sample(node_names, k=len(node_names))
        self.node_mapping = dict(zip(node_names, shuffled_names))

    def set_network_ground_truth(self, ground_truth):
        if self.node_mapping is None:
            self.setup_node_mapping(tuple(ground_truth.keys()))

        assert self.node_mapping.keys() == ground_truth.keys(), \
            "The mapping for node shuffling should match the ground truth"

        return super().set_network_ground_truth({k: ground_truth[self.node_mapping[k]] for k in ground_truth})


def create_shuffled_subclass(base_cls: Type[BioFuzzNet]):
    """Dynamically create a shuffled version of a BioFuzzNet sub-class
    If the base class is named BlaBlaSubBioNet, the new class is called ShuffledBlaBlaSubBioNet
    """
    shuffled_subclass = type(f"Shuffled{base_cls.__name__}", (LabelShuffleMixin, base_cls), {})
    return shuffled_subclass
