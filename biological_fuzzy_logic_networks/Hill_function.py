"""A Pytorch module defining a Hill function"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import torch

torch.set_default_tensor_type(torch.DoubleTensor)


class HillTransferFunction(torch.nn.Module):
    """Apply a Hill transformation on 1D input"""

    def __init__(self):
        """
        Initialise the parameters if the transfer functino
        """
        torch.nn.Module.__init__(self)

        self.n = torch.nn.Parameter(
            torch.normal(mean=0.3, std=0.4, size=(1,))
        )  # I add one afterwards in the forward function
        self.K = torch.nn.Parameter(torch.normal(mean=0, std=0.75, size=(1,)))

    def forward(self, x):
        """
        Tranforms a value through the transfer function
        Args:
            x = value to be transformed
        """
        # Hill function as in the Eduati paper
        K = torch.exp(self.K)
        n = 1 + torch.exp(self.n)  # Enforce n>1
        x = 1 - x
        output = (x**n) / (K**n + x**n)
        # in the normalized_Hill branch we normalize the outputs
        output = output * (1 + K**n)
        self.output_value = 1 - output
        return self.output_value
