##% Imports
import torch
# Set a seed
torch.manual_seed(2)
import argparse
import os
import pickle
from tqdm import tqdm

import bionics.biofuzznet.biofuzznet as biofuzznet


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input',
    type=str,
    required=True,
    help = "Relative path to the SIF file containing the network to use for simulation")
parser.add_argument('--outputfolder',
    type=str,
    required=True,
    help = "Relative ath to the folder where the simulation output will be saved. Should end with a backslash. Folder will be created if it does not exist")
parser.add_argument('--train',
    type=int,
    required=True,
    help = "Number of cells in the training set")
parser.add_argument('--test',
    type=int,
    required=True,
    help = "Number of cells in the test set")
parser.add_argument('--gates',
    type=str,
    required=True,
    help = "Gates of the network. If 'true', then uses the gates as specified in the input network file. If 'reverse' then reverses the gates compared to the network input file (AND become OR and OR become AND). If 'random', keeps the input network file structure but assigns the gates at random.")
args = parser.parse_args()

# Create the output folder
if not os.path.exists(args.outputfolder):
    os.mkdir(args.outputfolder)
    print(f"Directory {args.outputfolder} did not exist and was created")
else:
    print(f"Directory {args.outputfolder} already exists. File will be overwritten")

# Generate the data using a continuous simulation
network_true = biofuzznet.BioFuzzNet.build_BioFuzzNet_from_file(args.input)
network_true.initialise_random_truth_and_output(args.train + args.test)
# This is outside of the fold loop
# Thus we use the same transfer function parameters for all folds
# And just change the gates

for fold in tqdm(range(10)):
    if not os.path.exists(f"{args.outputfolder}/CV_{fold}"):
        os.mkdir(f"{args.outputfolder}/CV_{fold}")
        print(f"Directory {args.outputfolder}+/CV_{fold} did not exist and was created")
    else:
        print(f"Directory {args.outputfolder}+/CV_{fold} already exists. File will be overwritten")

    
    # Take care of the gates
    if args.gates == "reverse":
        # Change all OR gates to AND gates and vice versa
        for node, attributes in network_true.nodes(data=True):
            if attributes["node_type"] == "logic_gate_AND":
                attributes["node_type"] = "logic_gate_OR"
            elif attributes["node_type"] == "logic_gate_OR":
                attributes["node_type"] = "logic_gate_AND"
        print("Gates were reversed.")
    elif args.gates == "random":
        # Change all OR gates to AND gates and vice versa
        for node, attributes in network_true.nodes(data=True):
            if attributes["node_type"] in ["logic_gate_AND", "logic_gate_OR"]:
                coin_toss = torch.bernoulli(torch.tensor(0.5))
                if coin_toss:
                    attributes["node_type"] = "logic_gate_OR"
                else:
                    attributes["node_type"] = "logic_gate_AND"
        print("Gates were randomly assigned.")
        pickle.dump(network_true, open(f"{args.outputfolder}/CV_{fold}/model_structure_data.p", "wb"))
    elif args.gates == "true":
        print("Gates were not modified")
    else:
        print("UNKNOWN OPTION. Gates will not be modified.")
        
        

    # Set ground truth for root nodes
    for n in network_true.root_nodes:
        network_true.nodes()[n]["output_state"] = network_true.nodes()[n][
            "ground_truth"
        ]
    inputs = {
        n: network_true.nodes()[n]["output_state"] for n in network_true.root_nodes
    }
    # Simulate
    network_true.sequential_update(network_true.root_nodes)
    # The result of the simulation will be our ground truth
    ground_truth = {
        n: network_true.nodes()[n]["output_state"]
        for n in network_true.biological_nodes
    }

    # Define a train and test set
    train_input = {key: val[0:args.train] for key, val in inputs.items()}
    test_input = {key: val[args.train:args.train+args.test] for key, val in inputs.items()}

    train_ground_truth = {key: val[0:args.train] for key, val in ground_truth.items()}
    test_ground_truth = {key: val[args.train:args.train+args.test] for key, val in ground_truth.items()}

    # Save the datasets
    pickle.dump(train_input, open(f"{args.outputfolder}/CV_{fold}/train_input.p", "wb"))
    pickle.dump(test_input, open(f"{args.outputfolder}/CV_{fold}/test_input.p", "wb"))
    pickle.dump(train_ground_truth, open(f"{args.outputfolder}/CV_{fold}/train_ground_truth.p", "wb"))
    pickle.dump(test_ground_truth, open(f"{args.outputfolder}/CV_{fold}/test_ground_truth.p", "wb"))