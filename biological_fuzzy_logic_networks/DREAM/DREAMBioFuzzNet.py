import copy
import warnings
from datetime import datetime
from typing import Optional

import networkx as nx
import pandas as pd
import torch as torch
from tqdm.autonotebook import tqdm

from biological_fuzzy_logic_networks.DREAM.DREAMdataset import DREAMBioFuzzDataset
from biological_fuzzy_logic_networks.biofuzznet import BioFuzzNet
from biological_fuzzy_logic_networks.biomixnet import BioMixNet
from biological_fuzzy_logic_networks.utils import MSE_loss, read_sif
from biological_fuzzy_logic_networks.utils import has_cycle


class DREAMMixIn:
    # Setter Methods

    def initialise_random_truth_and_output(self, batch_size, to_cuda: bool = False):
        """
        Initialises the network so that the output_state and ground_truth are set to random tensors.
        Args:
            - batch_size: size of the tensor. All tensors will have the same size.
        NB: This is useful because output_state and ground_truth are set to None when adding nodes using self.add_fuzzy_node()
            and having None values creates unwanted behavior when using mathematical operations (NaN propagates to non-NaN tensors)
        """
        for node_name in self.nodes():
            node = self.nodes()[node_name]
            if node["node_type"] == "biological":
                node["ground_truth"] = torch.rand(batch_size)
                node["output_state"] = torch.rand(batch_size)
            else:
                node["output_state"] = torch.rand(batch_size)

            if to_cuda:
                node["output_state"] = node["output_state"].to("cuda:0")
                if node["node_type"] == "biological":
                    node["ground_truth"] = node["ground_truth"].to("cuda:0")

    def update_fuzzy_node(self, node: str, inhibition, to_cuda: bool = False) -> None:
        """
        A wrapper to call the correct updating function depending on the type of the node.
        Args:
            - node: name of the node to update
        """

        node_type = self.nodes()[node]["node_type"]
        if node_type == "biological":
            self.nodes()[node]["output_state"] = self.update_biological_node(
                node=node, inhibition=inhibition
            )
        else:
            self.nodes()[node]["output_state"] = self.integrate_logical_node(
                node=node, inhibition=inhibition, to_cuda=to_cuda
            )

    def update_biological_node(self, node: str, inhibition) -> torch.Tensor:
        """
        Returns the updated output state of a node when propagating  through the graph.
        Args:
            - node: name of the biological node to update
        Return:
            - a torch.Tensor representing the updated value of the node
        """
        parent_node = [p for p in self.predecessors(node)]
        if len(parent_node) > 1:
            raise AssertionError("This biological node has more than one incoming edge")
        elif len(parent_node) == 1:
            # The state of a root node stays the same
            return self.propagate_along_edge(
                edge=(parent_node[0], node), inhibition=inhibition
            )
        else:  # For a root edge
            return self.nodes()[node]["ground_truth"]

    def integrate_logical_node(
            self, node: str, inhibition, to_cuda: bool = False
    ) -> torch.Tensor:
        """
        A wrapper around integrate_NOT, integrate_OR and integrate_AND to integrate the values
        at any logical node independently of the gate.

        Args:
            - node: the name of the node representing the logical gate
        Returns:
            - The state at the logical gate after integration

        """
        if self.nodes[node]["node_type"] == "logic_gate_AND":
            return self.integrate_AND(node=node, inhibition=inhibition)
        if self.nodes[node]["node_type"] == "logic_gate_OR":
            return self.integrate_OR(node=node, inhibition=inhibition)
        if self.nodes[node]["node_type"] == "logic_gate_NOT":
            return self.integrate_NOT(node=node, inhibition=inhibition, to_cuda=to_cuda)
        else:
            raise NameError("This node is not a known logic gate.")

    def integrate_NOT(
            self, node: str, inhibition, to_cuda: bool = False
    ) -> torch.Tensor:
        """
        Computes the NOT operation at a NOT gate

        Args:
            node: the name of the node representing the NOT gate
        Returns:
            The output state at the NOT gate after computation
        """
        upstream_edges = [(pred, node) for pred in self.predecessors(node)]
        if len(upstream_edges) > 1:
            raise AssertionError("This NOT gate has more than one predecessor")
        if len(upstream_edges) == 0:
            raise AssertionError("This NOT gate has no predecessor")
        else:
            state_to_integrate = self.propagate_along_edge(
                edge=upstream_edges[0], inhibition=inhibition
            )
            ones = torch.ones(state_to_integrate.size())

            if to_cuda:
                ones = ones.to("cuda:0")
            # We work with tensors
            return ones - state_to_integrate

    def integrate_AND(self, inhibition, node: str) -> torch.Tensor:
        """
        Integrate the state values from all incoming nodes at an AND gate.
        Cannot support more than two input gates.

        Args:
            node: the name of the node representing the AND gate
        Returns:
            The output state at the AND gate after integration
        """
        upstream_edges = [(pred, node) for pred in self.predecessors(node)]
        if len(upstream_edges) > 2:
            raise AssertionError(
                f"The AND gate {node} has more than two incoming edges."
            )
        states_to_integrate = [
            self.propagate_along_edge(edge=edge, inhibition=inhibition)
            for edge in upstream_edges
        ]
        # Multiply all the tensors
        return states_to_integrate[0] * states_to_integrate[1]

    def integrate_OR(self, inhibition, node: str) -> torch.Tensor:
        """
        Integrate the state values from all incoming nodes at an OR gate.
        Cannot support more than two input gates.

        Args:
            node: the name of the node representing the OR gate
        Returns:
            The state at the OR gate after integration
        """
        upstream_edges = [(pred, node) for pred in self.predecessors(node)]
        if len(upstream_edges) > 2:
            raise AssertionError(
                f"The OR gate {node} has more than two incoming edges."
            )
        states_to_integrate = [
            self.propagate_along_edge(edge=edge, inhibition=inhibition)
            for edge in upstream_edges
        ]

        # Multiply all the tensors
        return (
                states_to_integrate[0]
                + states_to_integrate[1]
                - states_to_integrate[0] * states_to_integrate[1]
        )

    def propagate_along_edge(self, edge: tuple, inhibition) -> torch.Tensor:
        """
        Transmits node state along an edge.
        If an edge is simple: then it returns the state at the upstream node. No computation occurs in this case.
        If an edge sports a transfer function: then it computes the transfer function and returns the transformed state.

        Args:
            edge: The edge along which to propagate the state
        Returns:
            The new state at the target node of the edge
        """
        if edge not in self.edges():
            raise NameError(f"The input edge {edge} does not exist.")
            assert False
        elif self.edges()[edge]["edge_type"] == "simple":
            state_to_propagate = self.nodes[edge[0]]["output_state"]
            return state_to_propagate
        elif self.edges()[edge]["edge_type"] == "transfer_function":
            # The preceding state has to go through the Hill layer
            state_to_propagate = self.edges()[edge]["layer"](
                self.nodes[edge[0]]["output_state"]
            )
        else:
            NameError("The node type is incorrect")
            assert False

        if self.nodes[edge[0]]["node_type"] == "biological":
            state_to_propagate = state_to_propagate / inhibition[edge[0]]
        return state_to_propagate

    def sequential_update(
            self, input_nodes, inhibition, convergence_check=False, to_cuda: bool = False
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
            while len(non_updated_nodes) > 0:
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
                    if len(non_updated_parents) > 0:
                        for p in non_updated_parents:
                            # curr_nodes is FIFO: we first append the parents then the child
                            current_nodes.append(p)
                        current_nodes.append(curr_node)
                    # If all parents are updated, then we update
                    else:
                        self.update_fuzzy_node(curr_node, inhibition, to_cuda=to_cuda)
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
            length = 20  # 3 * max([len(cycle) for cycle in has_cycle(self)[1]])
            # We simulate length times then output the mean of the last length simulations
            # CHANGED length to int(length/2)
            states[0] = self.output_states
            for i in range(1, int(length)):
                states[i] = self.update_one_timestep_cyclic_network(
                    input_nodes,
                    inhibition,
                    loop_status,
                    convergence_check,
                    to_cuda=to_cuda,
                )
            last_states = {}
            for i in range(int(length)):
                states[length + i] = self.update_one_timestep_cyclic_network(
                    input_nodes,
                    inhibition,
                    loop_status,
                    convergence_check,
                    to_cuda=to_cuda,
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
            self,
            input_nodes,
            inhibition,
            loop_status,
            convergence_check=False,
            to_cuda: bool = False,
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
        while len(non_updated_nodes) > 0:
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
                if len(non_updated_parents) > 0:
                    for p in non_updated_parents:
                        # Check if there is a loop to which both the parent and the current node belong
                        for cycle in loop_status[1]:
                            if curr_node in cycle and p in cycle:
                                # Then we will need to update curr_node without updating its parent
                                non_updated_parents.remove(p)
                                break
                    # Now non_updated_parents only contains parents that are not part of a loop to which curr_node belongs
                    if len(non_updated_parents) > 0:
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
                    # print(curr_node)
                    self.update_fuzzy_node(curr_node, inhibition, to_cuda=to_cuda)
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
            logger=None,
            convergence_check: bool = False,
            save_checkpoint: bool = True,
            checkpoint_path: str = None,
            tensors_to_cuda: bool = False,
            patience: int = 5,
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
        if len(self.root_nodes) == 0:
            input_nodes = [k for k in valid_input.keys()]
            print(f"There were no root nodes, {input_nodes} were used as input")
        else:
            input_nodes = self.root_nodes

        # Instantiate the dataset
        # print(input)
        # print(ground_truth)
        # print(train_inhibitors)

        if tensors_to_cuda:
            for node_key, node_tensor in input.items():
                input[node_key] = node_tensor.to("cuda:0")
            for node_key, node_tensor in valid_input.items():
                valid_input[node_key] = node_tensor.to("cuda:0")
            for node_key, node_tensor in ground_truth.items():
                ground_truth[node_key] = node_tensor.to("cuda:0")
            for node_key, node_tensor in valid_ground_truth.items():
                valid_ground_truth[node_key] = node_tensor.to("cuda:0")
            for node_key, node_tensor in train_inhibitors.items():
                train_inhibitors[node_key] = node_tensor.to("cuda:0")
            for node_key, node_tensor in valid_inhibitors.items():
                valid_inhibitors[node_key] = node_tensor.to("cuda:0")

            # Transfer edges (model) to cuda
            for edge in self.transfer_edges:
                self.edges()[edge]["layer"].to("cuda:0")

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
        early_stopping_count = 0

        epoch_pbar = tqdm(range(epochs), desc="Loss=?.??e??")
        train_loss_running_mean = None
        for e in epoch_pbar:
            # Instantiate the model
            self.initialise_random_truth_and_output(batch_size, to_cuda=tensors_to_cuda)

            for X_batch, y_batch, inhibited_batch in tqdm(
                dataloader, desc="batch", total=len(dataset) // batch_size
            ):
                # In this case we do not use X_batch explicitly, as we just need the ground truth state of each node.
                # Reinitialise the network at the right size
                batch_keys = list(X_batch.keys())
                self.initialise_random_truth_and_output(
                    len(X_batch[batch_keys.pop()]), to_cuda=tensors_to_cuda
                )
                # predict and compute the loss
                self.set_network_ground_truth(ground_truth=y_batch)
                # Simulate
                loop_states = self.sequential_update(
                    input_nodes,
                    inhibited_batch,
                    convergence_check=convergence_check,
                    to_cuda=tensors_to_cuda,
                )

                # Get the predictions
                predictions = {
                    k: v for k, v in self.output_states.items() if k not in input_nodes
                }
                # predictions = self.output_states
                labels = {k: v for k, v in y_batch.items() if k in predictions}

                loss = MSE_loss(predictions=predictions, ground_truth=labels)

                # First reset then compute the gradients
                optim.zero_grad()
                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_value_(parameters, clip_value=0.5)
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1)
                # Update the parameters
                optim.step()
                # We save metrics with their time to be able to compare training vs validation
                # even though they are not logged with the same frequency
                if logger is not None:
                    logger.log_metric("train_loss", loss.detach().item())
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
                if train_loss_running_mean is not None: 
                    train_loss_running_mean = 0.1*loss.detach().item() + 0.9*train_loss_running_mean
                else:
                    train_loss_running_mean = loss.detach().item()
                epoch_pbar.set_description(f"Loss:{train_loss_running_mean:.2e}")
            # Validation
            with torch.no_grad():
                # Instantiate the model
                self.initialise_random_truth_and_output(
                    len(
                        valid_ground_truth[input_nodes[0]],
                    ),
                    to_cuda=tensors_to_cuda,
                )
                self.set_network_ground_truth(ground_truth=valid_ground_truth)
                # Simulation
                self.sequential_update(
                    input_nodes, valid_inhibitors, to_cuda=tensors_to_cuda
                )
                # Get the predictions
<<<<<<< HEAD
                predictions = {k: v for k, v in self.output_states.items() if k not in input_nodes}
                labels = {k: v for k, v in valid_ground_truth.items() if k in predictions}
                # predictions = self.output_states
                valid_loss = MSE_loss(
                    predictions=predictions, ground_truth=labels
                )
=======
                predictions = {
                    k: v for k, v in self.output_states.items() if k not in input_nodes
                }
                labels = {
                    k: v for k, v in valid_ground_truth.items() if k in predictions
                }
                # predictions = self.output_states
                valid_loss = MSE_loss(predictions=predictions, ground_truth=labels)
>>>>>>> b607f17 (Trial for larger synthetic perturbation)

                # No need to detach since there are no gradients
                if logger is not None:
                    logger.log_metric("valid_loss", valid_loss.item())

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

                if curr_best_val_loss > valid_loss:
                    early_stopping_count = 0
                    curr_best_val_loss = valid_loss
                    if checkpoint_path is not None:
                        module_of_edges = torch.nn.ModuleDict(
                            {
                                f"{edge[0]}@@@{edge[1]}": self.edges()[edge]["layer"]
                                for edge in self.transfer_edges
                            }
                        )

                        best_model_state = module_of_edges.state_dict()
                        best_optimizer_state = optim.state_dict()

                        # torch.save(
                        #     {
                        #         "epoch": e,
                        #         "model_state_dict": best_model_state,
                        #         "optimizer_state_dict": best_optimizer_state,
                        #         "loss": valid_loss,
                        #     },
                        #     f"{checkpoint_path}model.pt",
                        # )

                        # pred_df = pd.DataFrame(
                        #     {k: v.numpy() for k, v in predictions.items()}
                        # )
                        # pred_df.to_csv(f"{checkpoint_path}predictions_with_model.csv")
                else:
                    early_stopping_count += 1

                    if early_stopping_count > patience:
                        print("Early stopping")
                        if checkpoint_path is not None:

<<<<<<< HEAD
=======
                        if checkpoint_path is not None:
>>>>>>> b607f17 (Trial for larger synthetic perturbation)
                            torch.save(
                                {
                                    "epoch": e,
                                    "model_state_dict": best_model_state,
                                    "optimizer_state_dict": best_optimizer_state,
                                    "loss": valid_loss,
                                },
                                f"{checkpoint_path}model.pt",
                            )
<<<<<<< HEAD
    
=======

>>>>>>> b607f17 (Trial for larger synthetic perturbation)
                            pred_df = pd.DataFrame(
                                {k: v.numpy() for k, v in predictions.items()}
                            )
                            pred_df.to_csv(
                                f"{checkpoint_path}predictions_with_model_early_stopping.csv"
                            )

                        if convergence_check:
                            return losses, curr_best_val_loss, loop_states
                        else:
                            return losses, curr_best_val_loss, None
            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": e,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": best_optimizer_state,
                        "loss": valid_loss,
                    },
                    f"{checkpoint_path}model.pt",
                )
<<<<<<< HEAD
    
=======

>>>>>>> b607f17 (Trial for larger synthetic perturbation)
                pred_df = pd.DataFrame({k: v.numpy() for k, v in predictions.items()})
                pred_df.to_csv(f"{checkpoint_path}predictions_with_model_save.csv")

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

    @classmethod
    def build_DREAMBioFuzzNet_from_file(cls, filepath: str):
        """
        An alternate constructor to build the BioFuzzNet from the sif file instead of the lists of nodes and edges.
        AND gates should already be specified in the sif file, and should be named node1_and_node2 where node1 and node2 are the incoming nodes


        Args:
            - filepath: SIF file in tsv format [node1 edge_weight node2] if the network topology is contained in a file.
                If the file ha the format [node1 node2 edge_weight], then it can be converted in the desired format using  utils.change_SIF_convention

        """
        nodes, edges = read_sif(filepath)
        return DREAMBioFuzzNet(nodes, edges)


class DREAMBioMixNet(DREAMMixIn, BioMixNet):
    def __init__(self, nodes=None, edges=None):
        super(DREAMBioMixNet, self).__init__(nodes, edges)

    @classmethod
    def build_DREAMBioMixNet_from_file(cls, filepath: str):
        """
        An alternate constructor to build the BioFuzzNet from the sif file instead of the lists of nodes and edges.
        AND gates should already be specified in the sif file, and should be named node1_and_node2 where node1 and node2 are the incoming nodes


        Args:
            - filepath: SIF file in tsv format [node1 edge_weight node2] if the network topology is contained in a file.
                If the file ha the format [node1 node2 edge_weight], then it can be converted in the desired format using  utils.change_SIF_convention

        """
        nodes, edges = read_sif(filepath)
        return DREAMBioMixNet(nodes, edges)
