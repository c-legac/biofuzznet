"""
    This generate the a continuous version where absent domains are encoded as 0 and present domains can be encoded from 0.1 to 1
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
from tqdm import tqdm
from datetime import datetime
import pickle

import bionics.biofuzznet.biofuzznet as biofuzznet
import bionics.biofuzznet.biomixnet as biomixnet
import bionics.biofuzznet.biofuzzdataset as biofuzzdataset
import bionics.biofuzznet.Hill_function as Hill_function
import bionics.biofuzznet.utils as utils

import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model',
    type=str,
    required=True,
    help = "Path to input model")
parser.add_argument('--foldnum',
    type=int,
    required=True,
    help = "Number of folds for the MC simulation")
parser.add_argument('--outputfolder',
    type=str,
    required=True,
    help = "Path to output")
parser.add_argument('--epochs',
    type=int,
    required=True,
    help = "Number of epochs for optimisation")
parser.add_argument('--learningrate',
    type=float,
    required=True,
    help = "Learning rate for optimisation")
parser.add_argument('--batchsize',
    type=int,
    required=True,
    help = "Batch size for optimisation")
parser.add_argument('--rounds',
    type=int,
    required=True,
    help = "Number of rounds for optimisation. The model is optimised for a total of epochs*round epochs")
args = parser.parse_args()
# See if output folder exists
# Create the output folder
if not os.path.exists(args.outputfolder):
    os.mkdir(args.outputfolder)
    print(f"Directory {args.outputfolder} did not exist and was created")
else:
    print(f"Directory {args.outputfolder} already exists. File will be overwritten")


# Set a seed
torch.manual_seed(2)
# Set a default dtype
torch.set_default_tensor_type(torch.DoubleTensor)
# Root nodes are initialised to 0 or 1
# Use the predefined random states for generating the results
network = biofuzznet.BioFuzzNet.build_BioFuzzNet_from_file(args.model)
network.initialise_random_truth_and_output(4500)

# I used to have a random input
# Now let us prepare something similar to what we wanted to do for the CART
# ie have all vectors with 1 of 5 inputs
# then there are 10 vectors with 2 inputs
# choose 5 and try to predict the 5 other
# Let us put 300 cells for each input
# As test inputs I will use 11000, 00110, 00011, 10100, 01001
# The order is 41BB, CD28, CD40, CTLA4, IL15RA, CD45
# I also have to provide for CD45 which will be always set to 1 and will be the last bit
cell_inputs = [
    [1e-9, 1e-9, 1e-9, 1e-9, 1, 1],
    [1e-9, 1e-9, 1e-9, 1, 1e-9, 1],
    [1e-9, 1e-9, 1, 1e-9, 1e-9, 1],
    [1e-9, 1, 1e-9, 1e-9, 1e-9, 1],
    [1, 1e-9, 1e-9, 1e-9, 1e-9, 1],
    [1, 1e-9, 1e-9, 1e-9, 1, 1],
    [1e-9, 1, 1e-9, 1e-9, 1, 1],
    [1e-9, 1e-9, 1, 1e-9, 1, 1],
    [1e-9, 1e-9, 1e-9, 1, 1, 1],
    [1, 1e-9, 1e-9, 1, 1e-9, 1],
    [1e-9, 1, 1e-9, 1, 1e-9, 1],
    [1e-9, 1e-9, 1, 1, 1e-9, 1],
    [1, 1e-9, 1, 1e-9, 1e-9, 1],
    [1e-9, 1, 1, 1e-9, 1e-9, 1],
    [1, 1, 1e-9, 1e-9, 1e-9, 1],
]

# Generate the data
simulated_data = pd.DataFrame(columns = network.biological_nodes)
for combinations in range(len(cell_inputs)):
    for cell in tqdm(range(300)):
        input_dic = {"41BB": torch.tensor(cell_inputs[combinations][0]- 0.2*torch.rand(1)*cell_inputs[combinations][0]) + 0.2*torch.rand(1)*(1-cell_inputs[combinations][0]),
        "CD28":  torch.tensor(cell_inputs[combinations][1]- 0.2*torch.rand(1)*cell_inputs[combinations][1]) + 0.2*torch.rand(1)*(1-cell_inputs[combinations][1]),
        "CD40":  torch.tensor(cell_inputs[combinations][2]- 0.2*torch.rand(1)*cell_inputs[combinations][2])+ 0.2*torch.rand(1)*(1-cell_inputs[combinations][2]),
        "CTLA4":  torch.tensor(cell_inputs[combinations][3]- 0.2*torch.rand(1)*cell_inputs[combinations][3])+ 0.2*torch.rand(1)*(1-cell_inputs[combinations][3]),
        "IL15Ra":  torch.tensor(cell_inputs[combinations][4]- 0.2*torch.rand(1)*cell_inputs[combinations][4])+ 0.2*torch.rand(1)*(1-cell_inputs[combinations][4]),
        "CD45":  torch.tensor(cell_inputs[combinations][5]- 0.2*torch.rand(1)*cell_inputs[combinations][5])+ 0.2*torch.rand(1)*(1-cell_inputs[combinations][5])
        }
        network.set_network_ground_truth(input_dic)
        network.sequential_update(list(input_dic.keys()))
        output_states = network.output_states
        simulated_data = simulated_data.append(pd.Series({node: output_states[node].item() for node in output_states.keys()}), ignore_index=True)
input_col = list(input_dic.keys())
simulated_data.to_csv("simulated_data.csv")

# This is ugly but solves a type Error on np arrays
simulated_data = pd.read_csv("simulated_data.csv", index_col = 0)
simulated_data

# Initialise results dataframe
RMSE_df = pd.DataFrame(columns = simulated_data.columns)
test_set_df = pd.DataFrame(columns = ["test_1", "test_2", "test_3", "test_4", "test_5"])
params_df = pd.DataFrame(utils.obtain_params(network)[0]) # The true parameters 

rng =np.random.default_rng(2)
for fold in range(args.foldnum):
    # Re-initialize the network
    network = biofuzznet.BioFuzzNet.build_BioFuzzNet_from_file(args.model)
    core_train = [0,1,2,3,4]
    train_rows = list(rng.choice([i for i in range(5,15) ], 5, replace = False))
    test_rows = [i for i in range(5,15) if i not in train_rows and i not in [2]]
    train_rows = train_rows + core_train
    train_cells = []
    test_cells = []
    for index in train_rows:
        train_cells += [cell for cell in range(index, index + 300)]
    for index in test_rows:
        test_cells += [cell for cell in range(index, index + 300)]
    GT_train = {c : torch.tensor(simulated_data.loc[train_cells,c].values) for c in simulated_data.columns}
    GT_test = {c : torch.tensor(simulated_data.loc[test_cells,c].values) for c in simulated_data.columns}
    input_train = {c: GT_train[c] for c in input_col}
    input_test = {c: GT_test[c] for c in input_col}

    # Optimise
    try:
        all_losses = []
        losses = []
        loss= []
        test_loss = []
        for i in range(args.rounds):
            losses_i = network.conduct_optimisation(
                input=input_train,
                ground_truth=GT_train,
                test_input=input_test,
                test_ground_truth=GT_test,
                epochs=args.epochs,
                learning_rate=args.learningrate,
                batch_size=args.batchsize,
            )
            losses.append(losses_i)
            #plot_loss = sns.lineplot(data=losses_i, x="time", y="loss", hue="phase")
            #plot_fig = plot_loss.get_figure()
            #plot_fig.savefig(f"{args.outputfolder}/intermediate_loss_{i}.png")


        # Concatenate the losses of the iterative runs
        all_losses = pd.DataFrame()
        all_losses = pd.concat(losses)
        all_losses.reset_index(inplace=True)
        all_losses


        # Plot the losses to check optimisation happened correctly
        plot_loss = sns.lineplot(data=all_losses, x="time", y="loss", hue="phase")
        plot_fig = plot_loss.get_figure()
        plot_fig.savefig(f"{args.outputfolder}/total_loss_CV{fold}.png")

        # Generate the measures that will be useful
        # Simulate the BFZ on the test set
        network.set_network_ground_truth(input_test)
        network.sequential_update(network.root_nodes)
        # Compute the test error and save it into the dataframe
        output_RMSE = utils.compute_RMSE_outputs(model = network, ground_truth=GT_test)
        RMSE = pd.Series(output_RMSE)
        RMSE_df = RMSE_df.append(RMSE, ignore_index = True)

        # Save the transfer edges parameters
        BFZ_params = utils.obtain_params(network)[0]
        params = pd.Series(BFZ_params)
        params_df = params_df.append(params, ignore_index = True)

        # Save the input combinations that we used for testing
        input_combi = pd.Series(test_rows)
        input_combi.index = ["test_1", "test_2", "test_3", "test_4", "test_5"]
        test_set_df = test_set_df.append(input_combi, ignore_index = True)

    except Exception as e:
        print(f"Fold {fold} was interrupted because of error {e}")
        with open(f"{args.outputfolder}/eror_fold_{fold}.txt", "w") as f:
            f.write(f"Fold {fold} failed because of exception")
            f.write("\n")
            f.write(f"{e}")
            f.write(f"The test rows of this fold were: {test_rows}")


    # Save the data we will need later
    print(fold)
    
with open(f"{args.outputfolder}/hyperparameters.txt", "w") as f:
    f.write(f"model: {args.model}, epochs: {args.epochs}, batchsize: {args.batchsize}, rounds: {args.rounds}, learningrate: {args.learningrate},folds: {args.foldnum}")
    f.close()
RMSE_df.to_csv(f"{args.outputfolder}/RMSE_df.csv")
params_df.to_csv(f"{args.outputfolder}/params_df.csv")
test_set_df.to_csv(f"{args.outputfolder}/test_set_df.csv")