##% Imports

import torch
# Set a seed
torch.manual_seed(2)
import argparse
import os
import pickle
import seaborn as sns
import pandas as pd

import bionics.biofuzznet.biofuzznet as biofuzznet
import bionics.biofuzznet.biomixnet as biomixnet
import bionics.biofuzznet.utils as utils


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--outputfolder',
    type=str,
    required=True,
    help = "Relative path to the folder where the optimisation output will be saved. Should end with a backslash. Folder will be created if it does not exist")
parser.add_argument('--folds',
    type=int,
    required=True,
    help = "Number of different runs to simulate")

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
args = parser.parse_args()

if not os.path.exists(args.outputfolder):
    os.mkdir(args.outputfolder)
    print(f"Directory {args.outputfolder} did not exist and was created")
else:
    print(f"Directory {args.outputfolder} already exists. File will be overwritten")
   

def create_pos_loop():
    pos_loop = biofuzznet.BioFuzzNet()

    pos_loop.add_fuzzy_node("A", "BIO")
    pos_loop.add_fuzzy_node("B", "BIO")
    pos_loop.add_fuzzy_node("C", "BIO")
    pos_loop.add_fuzzy_node("OR", "OR")

    pos_loop.add_transfer_edge("A", "OR")
    pos_loop.add_transfer_edge("B", "C")
    pos_loop.add_transfer_edge("C", "OR")
    pos_loop.add_simple_edge("OR", "B")
    return pos_loop

test_truth ={}
test_pred = {}
params_truth = {}
params_pred = {}
losses = {}

for fold in range(args.folds):
    print(f"CURRENTLY RUNNING FOLD {fold}")
    # Simulate data
    G_sim = create_pos_loop()
    train_vals_pos = {"A":torch.rand(1000), "B":torch.rand(1000), "C":torch.rand(1000)}
    G_sim.initialise_random_truth_and_output(1000)
    G_sim.set_network_ground_truth(train_vals_pos)
    for node in G_sim.biological_nodes:
        # Initialise everything to a random value
        G_sim.nodes()[node]["output_state"] = G_sim.nodes()[node]["ground_truth"] 
    ground_truth = G_sim.sequential_update(["A"], convergence_check = True)[15]

    test_vals_pos = {"A":torch.rand(100), "B":torch.rand(100), "C":torch.rand(100)}
    G_sim.initialise_random_truth_and_output(100)
    G_sim.set_network_ground_truth(test_vals_pos)
    for node in G_sim.biological_nodes:
        G_sim.nodes()[node]["output_state"] = G_sim.nodes()[node]["ground_truth"] 
    ground_truth_test = G_sim.sequential_update(["A"], convergence_check = True)[15]

    
    # Fit a loop
    G_optim = create_pos_loop()
    losses_fold = G_optim.conduct_optimisation(
        ground_truth=ground_truth,
        input = {"A": ground_truth["A"]},
        test_ground_truth=ground_truth_test,
        test_input= {"A": ground_truth_test["A"]},
        epochs = args.epochs,
        learning_rate=args.learningrate,
        batch_size = args.batchsize
    )

    # Save parameters
    losses[fold] = losses_fold
    test_truth[fold] = ground_truth_test
    test_pred[fold] = G_optim.output_states
    params_truth[fold] = utils.obtain_params(G_sim)
    params_pred[fold] = utils.obtain_params(G_optim)


pickle.dump(losses, open(f"{args.outputfolder}/losses.p", "wb"))
pickle.dump(test_truth, open(f"{args.outputfolder}/test_truth.p", "wb"))
pickle.dump(test_pred, open(f"{args.outputfolder}/test_pred.p", "wb"))
pickle.dump(params_truth, open(f"{args.outputfolder}/params_truth.p", "wb"))
pickle.dump(params_pred, open(f"{args.outputfolder}/params_pred.p", "wb"))



