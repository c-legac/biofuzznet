##% Imports
import torch
# Set a seed
torch.manual_seed(2000)
import argparse
import os
import pickle

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
pickle.dump(train_input, open(f"{args.outputfolder}train_input.p", "wb"))
pickle.dump(test_input, open(f"{args.outputfolder}test_input.p", "wb"))
pickle.dump(train_ground_truth, open(f"{args.outputfolder}train_ground_truth.p", "wb"))
pickle.dump(test_ground_truth, open(f"{args.outputfolder}test_ground_truth.p", "wb"))