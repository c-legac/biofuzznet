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
import bionics.biofuzznet.utils as utils


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--inputnetwork',
    type=str,
    required=True,
    help = "Relative path to the SIF file containing the network to optimise")
parser.add_argument('--inputdata',
    type=str,
    required=True,
    help = "Relative path to the folder containing the train and test data. Should end with a backslash")
parser.add_argument('--outputfolder',
    type=str,
    required=True,
    help = "Relative path to the folder where the optimisation output will be saved. Should end with a backslash. Folder will be created if it does not exist")
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

# Load data
network = biofuzznet.BioFuzzNet.build_BioFuzzNet_from_file(args.inputnetwork)
train_input = pickle.load(open(f"{args.inputdata}train_input.p", "rb"))
test_input = pickle.load(open(f"{args.inputdata}test_input.p", "rb"))
train_ground_truth = pickle.load(open(f"{args.inputdata}train_ground_truth.p", "rb"))
test_ground_truth = pickle.load(open(f"{args.inputdata}test_ground_truth.p", "rb"))

# See if output folder exists
# Create the output folder
if not os.path.exists(args.outputfolder):
    os.mkdir(args.outputfolder)
    print(f"Directory {args.outputfolder} did not exist and was created")
else:
    print(f"Directory {args.outputfolder} already exists. File will be overwritten")
    
# Optimise
losses = []
for i in range(args.rounds):
    losses_i = network.conduct_optimisation(
        input=train_input,
        ground_truth=train_ground_truth,
        test_input=test_input,
        test_ground_truth=test_ground_truth,
        epochs=args.epochs,
        learning_rate=args.learningrate,
        batch_size=args.batchsize,
    )
    losses.append(losses_i)
    plot_loss = sns.lineplot(data=losses_i, x="time", y="loss", hue="phase")
    plot_fig = plot_loss.get_figure()
    plot_fig.savefig(f"{args.outputfolder}/intermediate_loss_{i}.png")


# Concatenate the losses of the iterative runs
all_losses = pd.concat(losses)
all_losses.reset_index(inplace=True)
all_losses

# Plot the losses to check optimisation happened correctly
plot_loss = sns.lineplot(data=all_losses, x="time", y="loss", hue="phase")
plot_fig = plot_loss.get_figure()
plot_fig.savefig(f"{args.outputfolder}/total_loss.png")

# Save the data we will need later
parameters_opt, n_opt, K_opt = utils.obtain_params(network)
pickle.dump(parameters_opt, open(f"{args.outputfolder}/parameters_opt.p", "wb"))
pickle.dump(network, open(f"{args.outputfolder}/model.p", "wb"))
pickle.dump(all_losses, open(f"{args.outputfolder}/losses.p", "wb"))

with open(f"{args.outputfolder}/hyperparameters.txt", "w") as f:
    f.write(f"model: {args.model}, epochs: {args.epochs}, batchsize: {args.batchsize}, rounds: {args.rounds}, learningrate: {args.learningrate}")
    f.close()