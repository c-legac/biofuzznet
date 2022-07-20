__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import torch

torch.set_default_tensor_type(torch.DoubleTensor)


class MixedGate(torch.nn.Module):

    def __init__(self, AND_param: float, AND_function, OR_function) -> None:
        
        torch.nn.Module.__init__(self)
        self.AND_param = torch.nn.Parameter(torch.tensor(AND_param))
        self.AND_function = AND_function
        self.OR_function = OR_function
        self.output_value = None

    def forward(self, x):
        # x is the node at which we integrate the inputs
        AND_value = self.AND_function(x)
        OR_value = self.OR_function(x)
        output = torch.sigmoid(self.AND_param) * AND_value + (1 - torch.sigmoid(self.AND_param) )* OR_value
        self.output_value = output
        return output

    