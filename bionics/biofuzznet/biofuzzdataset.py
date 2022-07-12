" Define the dataset structure to be sed with BioFuzzNets"

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import torch

# Create datasets and dataloader objects to iterate over batches of data


class BioFuzzDataset(torch.utils.data.Dataset):
    """We need the dataset to know which data vector belongs to which node"""

    def __init__(self, input_dict, ground_truth_dict):
        """Create the dataset.

        Args:
            input_dict: dictionnary mapping input nodes (keys) to tensors
                 representing the measured value at those nodes for several single cells.
            ground_truth_dict: dictionnary mapping biological nodes (keys) to tensors
                 representing the measured values at those nodes for several single cells.
        """
        self.X = input_dict
        self.y = ground_truth_dict

    def __getitem__(self, i):
        """For a given cell i, return two dictionnaries representing input and output values for a given cell.
        Args:
            i: int indentifying a datapoint (a single cell) in the dataset
        Returns:
            input_dict: dictionnary mapping input nodes (keys) to the measured value at those node for a given cell i.
            output_dict: dictionnary mapping biological nodes (keys) to the measured values at those nodes for a given cell i.
        """

        input_dict = {key: self.X[key][i] for key in self.X.keys()}
        output_dict = {key: self.y[key][i] for key in self.y.keys()}
        return input_dict, output_dict

    def __len__(self):
        """Return the number of single cells in the dataset."""
        keys = list(self.X.keys())
        k = keys.pop()
        # The number of different cells is the number of measurements at any input node.
        return len(self.X[k])
