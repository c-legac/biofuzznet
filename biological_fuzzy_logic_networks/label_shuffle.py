import random
from typing import Type

import torch

from biological_fuzzy_logic_networks.biofuzznet import BioFuzzNet


class LabelShuffleMixin:
    """A BFN mixin that remaps all biological nodes in the network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = list(self.biological_nodes)
        # random.sample is without replacement
        shuffled_names = random.sample(names, k=len(names))
        self.node_mapping = dict(zip(names, shuffled_names))

    def conduct_optimisation(
            self,
            input: dict,
            ground_truth: dict,
            test_input: dict,
            test_ground_truth: dict,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            optim_wrapper=torch.optim.Adam,
    ):
        all_data = dict(**input, **ground_truth)
        all_test_data = dict(**test_input, **test_ground_truth)

        new_input = {k: all_data[self.node_mapping[k]] for k in input}
        new_ground_truth = {k: all_data[self.node_mapping[k]] for k in ground_truth}
        new_test_input = {k: all_test_data[self.node_mapping[k]] for k in test_input}
        new_test_ground_truth = {k: all_data[self.node_mapping[k]] for k in test_ground_truth}

        return super(LabelShuffleMixin, self).conduct_optimisation(
            input=new_input,
            ground_truth=new_ground_truth,
            test_input=new_test_input,
            test_ground_truth=new_test_ground_truth,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optim_wrapper=optim_wrapper
        )


def create_shuffled_subclass(base_cls: Type[BioFuzzNet]):
    """Dynamically create a shuffled version of a BioFuzzNet sub-class
    If the base class is named BlaBlaSubBioNet, the new class is called ShuffledBlaBlaSubBioNet
    """
    shuffled_subclass = type(f"Shuffled{base_cls.__name__}", (LabelShuffleMixin, base_cls), {})
    return shuffled_subclass

