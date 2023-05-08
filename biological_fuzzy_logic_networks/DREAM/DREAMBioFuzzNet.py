from biological_fuzzy_logic_networks.biofuzznet import BioFuzzNet
from biological_fuzzy_logic_networks.biomixnet import BioMixNet
from biological_fuzzy_logic_networks.utils import MSE_loss
from biological_fuzzy_logic_networks.DREAM.DREAMdataset import DREAMBioFuzzDataset
from biological_fuzzy_logic_networks.utils import has_cycle
import networkx as nx
from typing import Optional
import torch as torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import warnings
import copy


class DREAMMixIn:
    def update_fuzzy_node(self, node: str, inhibition) -> None:
        """
        A wrapper to call the correct updating function depending on the type of the node.
        Args:
            - node: name of the node to update
        """
        node_type = self.nodes()[node]["node_type"]
        if node_type == "biological":
            self.nodes()[node]["output_state"] = self.update_biological_node(node)
        else:
            self.nodes()[node]["output_state"] = self.integrate_logical_node(node)

        # Inhibit nodes by dividing by a certain factor, if not inhibited inhibition is 1
        self.nodes()[node]["output_state"] = (
            self.nodes()[node]["output_state"] / inhibition[node]
        )

    def sequential_update(
        self, input_nodes, inhibition, convergence_check=False
    ) -> Optional[dict]:
        """
        Update the graph by propagating the signal from root node (or given input node)
        to leaf node. This update is sequential according to Boolean networks terminology.

        Method overview:
            The graph is traversed from root node to leaf node.
            The list of the nodes to be updated is implemented as a queue in a First In First Out (FIFO)
                 in order to update parents before their children.

        Args:
            - input_nodes: Nodes for which the ground truth is known and used as input for simulation (usually root nodes)
            - convergence_check: default False. In case one wants to check convergence of the simulation
                 for a graph with a loop, this Boolean should be set to True, and output states of the model
                 over the course of the simulation will be saved and returned. This has however not been
                 optimised for memory usage. Use with caution.
        """
        if convergence_check:
            warnings.warn(
                "convergence_check has been set to True. All simulation states will be saved and returned. This has not been optimised for memory usage and is implemented in a naive manner. Proceed with caution."
            )
        states = {}
        loop_status = has_cycle(self)
        if not loop_status[0]:
            current_nodes = copy.deepcopy(input_nodes)
            non_updated_nodes = [n for n in self.nodes()]
            safeguard = 0
            node_number = len([n for n in self.nodes()])
            while non_updated_nodes != []:
                safeguard += 1
                if safeguard > 10 * node_number:
                    print(
                        "Safeguard activated at 10*total number of nodes repetitions. Check if your network has loops. If node augment the safeguard."
                    )
                    break

                # curr_nodes is FIFO
                curr_node = current_nodes.pop(0)
                # If the node has not yet been updated
                if curr_node in non_updated_nodes:
                    parents = [pred for pred in self.predecessors(curr_node)]
                    non_updated_parents = [p for p in parents if p in non_updated_nodes]
                    # If one parent is not updated yet, then we cannot update
                    if non_updated_parents != []:
                        for p in non_updated_parents:
                            # curr_nodes is FIFO: we first append the parents then the child
                            current_nodes.append(p)
                        current_nodes.append(curr_node)
                    # If all parents are updated, then we update
                    else:
                        self.update_fuzzy_node(curr_node, inhibition)
                        non_updated_nodes.remove(curr_node)
                        cont = True
                        while cont:
                            try:
                                current_nodes.remove(curr_node)
                            except ValueError:
                                cont = False
                        child_nodes = [c for c in self.successors(curr_node)]
                        for c in child_nodes:
                            if c in non_updated_nodes:
                                current_nodes.append(c)
        else:
            # The time of the simulation is 2 times the size of the biggest cycle
            length = 3 * max([len(cycle) for cycle in has_cycle(self)[1]])
            # We simulate length times then output the mean of the last length simulations
            # CHANGED length to int(length/2)
            states[0] = self.output_states
            for i in range(1, int(length)):
                states[i] = self.update_one_timestep_cyclic_network(
                    input_nodes, inhibition, loop_status, convergence_check
                )
            last_states = {}
            for i in range(int(length)):
                print(i)
                states[length + i] = self.update_one_timestep_cyclic_network(
                    input_nodes, inhibition, loop_status, convergence_check
                )
                last_states[i] = {
                    n: self.nodes()[n]["output_state"] for n in self.nodes()
                }
            # Set the output to the mean of the last steps

            for n in self.nodes():
                output_tensor = last_states[0][n]
                for i in range(int(length) - 1):
                    output_tensor = output_tensor + last_states[i + 1][n]
                output_tensor = output_tensor / length
                self.nodes()[n]["output_state"] = output_tensor
        if convergence_check:
            return states  # For checking convergence
        else:
            return None

    def update_one_timestep_cyclic_network(
        self, input_nodes, inhibition, loop_status, convergence_check=False
    ) -> Optional[dict]:
        """
        Does the sequential update of a directed cyclic graph over one timestep: ie updates each node in the network only once.
        Args:
            - input_nodes: the node to start updating from, ie those for which we give the ground truth as input to the model
            - loop_status: the value returned by utils.has_cycle(self) which is a tuple (bool, list) where bool is True if the
            graph has a directed cycle, and the list is the list of all directed cycles in the graph
            - convergence_check: default False. In case one wants to check convergence of the simulation
                 for a graph with a loop, this Boolean should be set to True, and output state of the one-step simulation will be saved and returned. This has however not been
                 optimised for time and memory usage. Use with caution.
        """
        if convergence_check:
            warnings.warn(
                "convergence_check has been set to True. All simulation states will be saved and returned. This has not been optimised for memory usage and is implemented in a naive manner. Proceed with caution."
            )

        current_nodes = copy.deepcopy(input_nodes)
        non_updated_nodes = [n for n in self.nodes()]
        while non_updated_nodes != []:
            # curr_nodes is a queue, hence FIFO (first in first out)
            # when popping the first item, we obtain the one that has been in the queue the longest
            curr_node = current_nodes.pop(0)
            # If the node has not yet been updated
            if curr_node in non_updated_nodes:
                can_update = False
                non_updated_parents = [
                    p for p in self.predecessors(curr_node) if p in non_updated_nodes
                ]
                # Check if parents are updated
                if non_updated_parents != []:
                    for p in non_updated_parents:
                        # Check if there is a loop to which both the parent and the current node belong
                        for cycle in loop_status[1]:
                            if curr_node in cycle and p in cycle:
                                # Then we will need to update curr_node without updating its parent
                                non_updated_parents.remove(p)
                                break
                    # Now non_updated_parents only contains parents that are not part of a loop to which curr_node belongs
                    if non_updated_parents != []:
                        can_update = False
                        for p in non_updated_parents:
                            current_nodes.append(p)
                    else:
                        can_update = True
                    # The parents that were removed will be updated later as they are still part of non_updated nodes
                else:  # If all node parents are updated then no problem
                    can_update = True
                if not can_update:
                    # Then we reappend the current visited node
                    current_nodes.append(curr_node)
                else:  # Here we can update
                    print(curr_node)
                    self.update_fuzzy_node(curr_node, inhibition)
                    non_updated_nodes.remove(curr_node)
                    cont = True
                    while cont:
                        try:
                            current_nodes.remove(curr_node)
                        except ValueError:
                            cont = False
                    child_nodes = [c for c in self.successors(curr_node)]
                    for c in child_nodes:
                        if c in non_updated_nodes:
                            current_nodes.append(c)
        if convergence_check:
            return self.output_states  # For checking convergence
        else:
            return None

    def conduct_optimisation(
        self,
        input: dict,
        ground_truth: dict,
        train_inhibitors: dict,
        valid_input: dict,
        valid_ground_truth: dict,
        valid_inhibitors: dict,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optim_wrapper=torch.optim.Adam,
        convergence_check: bool = False,
        save_checkpoint: bool = True,
        checkpoint_path: str = None,
    ):
        """
        The main function of this class.
        Optimise the tranfer function parameters in a FIXED topology with FIXED input gates.
        For the moment, the optimizer is ADAM and the loss function is the MSELoss over all observed nodes (see utils.MSE_Loss)
        Method overview:
            The graph states are updated by traversing the graph from root node to leaf node (forward pass).
            The transfer function parameters are then updated using backpropagation.
            The use of backpropagation forces the use of a sequential update scheme.

        Args:
            - input: dict of torch.Tensor mapping root nodes name to their input value
                (which is assumed to also be their ground truth value, otherwise those nodes will never be fitted correctly)
                It is assumed that every node in input is an input node that should be known to the model prior to simulation.
                Input nodes are then used as the start for the sequential update algorithm.
                input should usually contain the value at root nodes, but in the case where the graph contains a cycle,
                other nodes can be specified.
            - ground_truth: training dict of {node_name: torch.Tensor} mapping each observed biological node to its measured values
                Only  the nodes present in ground_truth will be used to compute the loss/
            - valid_input: dict of torch.Tensor containing root node names mapped to the input validation data
            - valid_ground_truth:  dict of torch.Tensor mapping node names to their value from the validation set
            - epochs: number of epochs for optimisation
            - batch_size: batch size for optimisation
            - learning_rate : learning rate for optimisation with ADAM
            - optim_wrapper: a wrapper function for the optimiser. It should take as argument:
                - the parameters to optimise
                - the learning rate

        POSSIBLE UPDATES:
            - Allow tuning between AND and OR gates using backpropagation
        """

        torch.autograd.set_detect_anomaly(True)
        torch.set_default_tensor_type(torch.DoubleTensor)
        # Input nodes
        if self.root_nodes == []:
            input_nodes = [k for k in valid_input.keys()]
            print(f"There were no root nodes, {input_nodes} were used as input")
        else:
            input_nodes = self.root_nodes

        # Instantiate the dataset
        # print(input)
        # print(ground_truth)
        # print(train_inhibitors)
        dataset = DREAMBioFuzzDataset(input, ground_truth, train_inhibitors)

        # Instantiate the dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Keep track of the parameters
        parameters = []
        for edge in self.transfer_edges:
            layer = self.edges()[edge]["layer"]
            parameters += [layer.n, layer.K]

        optim = optim_wrapper(parameters, learning_rate)

        # Train the model
        losses = pd.DataFrame(columns=["time", "loss", "phase"])
        curr_best_val_loss = 1e6

        for e in tqdm(range(epochs)):
            # Instantiate the model
            self.initialise_random_truth_and_output(batch_size)

            for X_batch, y_batch, inhibited_batch in dataloader:
                # In this case we do not use X_batch explicitly, as we just need the ground truth state of each node.
                # Reinitialise the network at the right size
                batch_keys = list(X_batch.keys())
                self.initialise_random_truth_and_output(len(X_batch[batch_keys.pop()]))
                # predict and compute the loss
                self.set_network_ground_truth(ground_truth=y_batch)
                # Simulate
                loop_states = self.sequential_update(
                    input_nodes, inhibited_batch, convergence_check=convergence_check
                )

                # Get the predictions
                predictions = self.output_states

                loss = MSE_loss(predictions=predictions, ground_truth=y_batch)

                # First reset then compute the gradients
                optim.zero_grad()
                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_value_(parameters, clip_value=5)
                # Update the parameters
                optim.step()
                # We save metrics with their time to be able to compare training vs validation even though they are not logged with the same frequency
                losses = pd.concat(
                    [
                        losses,
                        pd.DataFrame(
                            {
                                "time": datetime.now(),
                                "loss": loss.detach().item(),
                                "phase": "train",
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
            with torch.no_grad():
                # Instantiate the model
                self.initialise_random_truth_and_output(
                    len(valid_ground_truth[input_nodes[0]])
                )
                self.set_network_ground_truth(ground_truth=valid_ground_truth)
                # Simulation
                self.sequential_update(input_nodes, valid_inhibitors)
                # Get the predictions
                predictions = self.output_states
                valid_loss = MSE_loss(
                    predictions=predictions, ground_truth=valid_ground_truth
                )

                if curr_best_val_loss > valid_loss:
                    curr_best_val_loss = valid_loss
                    module_of_edges = torch.nn.ModuleDict(
                        {
                            f"{edge[0]}@@@{edge[1]}": self.edges()[edge]["layer"]
                            for edge in self.transfer_edges
                        }
                    )
                    torch.save(
                        {
                            "epoch": e,
                            "model_state_dict": module_of_edges.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "loss": valid_loss,
                        },
                        f"{checkpoint_path}model.pt",
                    )

                # No need to detach since there are no gradients
                losses = pd.concat(
                    [
                        losses,
                        pd.DataFrame(
                            {
                                "time": datetime.now(),
                                "loss": valid_loss.item(),
                                "phase": "valid",
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
        if convergence_check:
            return losses, curr_best_val_loss, loop_states
        else:
            return losses, curr_best_val_loss, None

    def load_from_checkpoint(self, model_state_dict):
        module_dict = torch.nn.ModuleDict(
            {
                f"{edge[0]}@@@{edge[1]}": self.edges()[edge]["layer"]
                for edge in self.transfer_edges
            }
        )
        module_dict.load_state_dict(model_state_dict)
        edge_att = {
            (k.split("@@@")[0], k.split("@@@")[1]): {"layer": v}
            for k, v in module_dict.items()
        }
        nx.set_edge_attributes(self, edge_att)


class DREAMBioFuzzNet(DREAMMixIn, BioFuzzNet):
    def __init__(self, nodes=None, edges=None):
        super(DREAMBioFuzzNet, self).__init__(nodes, edges)


class DREAMBioMixNet(DREAMMixIn, BioMixNet):
    def __init__(self, nodes=None, edges=None):
        super(DREAMBioMixNet, self).__init__(nodes, edges)
