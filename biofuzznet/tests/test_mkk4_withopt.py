from biofuzznet.utils import read_sif
from biofuzznet.biofuzznet import BioFuzzNet
import networkx as nx
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# torch.autograd.detect_anomaly(check_nan=False)

cl_data = pd.read_csv("/users/adr/Box/CAR_Tcells/Data/DREAMdata/ZR7530.csv")
non_marker_cols = ["treatment", "cell_line", "time", "cellID", "fileID"]
markers = [c for c in cl_data.columns if c not in non_marker_cols]

cl_data = cl_data[cl_data["time"] == 9]
scaler = MinMaxScaler()
cl_data[markers] = scaler.fit_transform(cl_data[markers])

cl_data

nodes1, edges1 = read_sif(
    "/users/adr/Box/CAR_Tcells/Data/DREAMdata/DREAM_PKN_original.sif"
)

nodes, edges = read_sif(
    "/users/adr/Box/CAR_Tcells/Data/DREAMdata/DREAM_PKN_for_BFZ_input.sif"
)
edges[("JNK", "p53")] = 1
sel_nodes = ["MKK4", "JNK", "p53", "RB"]

sel_edges = {
    k: v for k, v in edges.items() if (k[1] in sel_nodes) and (k[0] in sel_nodes)
}
sel_edges

G = nx.from_edgelist(sel_edges)

nx.draw(G, with_labels=True)

# Create an empty BioFuzzNet
my_model = BioFuzzNet(None, None)

# Add nodes
# my_model.add_fuzzy_node("input", "BIO")
my_model.add_fuzzy_node("p.MKK4", "BIO")  # A biological node
my_model.add_fuzzy_node("p.JNK", "BIO")
my_model.add_fuzzy_node("p.p53", "BIO")
my_model.add_fuzzy_node("p.RB", "BIO")
# Add edges
# my_model.add_transfer_edge("input","p.MKK4")
my_model.add_transfer_edge("p.MKK4", "p.JNK")
my_model.add_transfer_edge("p.JNK", "p.p53")
my_model.add_transfer_edge("p.p53", "p.RB")

my_model.root_nodes

sel_data = cl_data.loc[(cl_data["treatment"] == "EGF"), :][
    ["p.p53", "p.MKK4", "p.JNK", "p.RB"]
]
sel_data.reset_index(drop=True, inplace=True)
# sel_data[sel_data["p.MKK4"] == 0.0] = 1e-9
#  = sel_data[sel_data["p.MKK4"] != 0.0]

train, test = train_test_split(sel_data)
print(train.shape)
print(test.shape)
train = train.to_dict("list")
test = test.to_dict("list")

train = {k: torch.DoubleTensor(v) for k, v in train.items()}
test = {k: torch.DoubleTensor(v) for k, v in test.items()}

input_train = {"p.MKK4": train["p.MKK4"]}
input_test = {"p.MKK4": test["p.MKK4"]}

print(input_train["p.MKK4"].shape)
input_train

print(train["p.MKK4"].shape)
train

print(train["p.MKK4"])

# Now we just need to specify some optimisation parameters

learning_rate = 5e-3
epochs = 2  # That seems like a lot, but 1 epoch is basically one simulation of the network on all datapoints, so we need a lot of them
batch_size = 1

# try:
loss = my_model.conduct_optimisation(
    input=input_train,
    test_input=input_test,
    ground_truth=train,
    epochs=epochs,
    test_ground_truth=test,
    learning_rate=learning_rate,
    batch_size=batch_size,
)
# except:
#     print("optimisation failed")


# class HillTransferFunctiontest(torch.nn.Module):
#     """Apply a Hill transformation on 1D input"""

#     def __init__(self, K, n):
#         """
#         Initialise the parameters if the transfer functino
#         """
#         torch.nn.Module.__init__(self)

#         self.n = torch.nn.Parameter(n)
#         self.K = torch.nn.Parameter(K)

#     def forward(self, x):
#         """
#         Tranforms a value through the transfer function
#         Args:
#             x = value to be transformed
#         """
#         # I want to constrain K and n to be positive
#         # Hence I'll feed them into an exponential
#         output = (x ** torch.exp(self.n)) / (
#             torch.exp(self.K) ** torch.exp(self.n) + x ** torch.exp(self.n)
#         )
#         self.output_value = output
#         return output


# ntest = [2.7571, 0.2072, 1.3665, 0.4239, 0.5019, 0.9385]
# K_test = [0.4412, 0.4821, 0.0998, 0.6090, 0.2535, 0.4413]

# ntestraw = np.log(ntest)


# def inversig(y):
#     return np.log(y / (1 - y))


# Ktestraw = [inversig(y) for y in K_test]

# print(ntestraw)
# print(Ktestraw)

# input = train["p.MKK4"]

# n = ntestraw[3:]
# K = Ktestraw[3:]
# lay1 = HillTransferFunctiontest(torch.tensor(n[0]), torch.tensor(K[0]))
# lay2 = HillTransferFunctiontest(torch.tensor(n[1]), torch.tensor(K[1]))
# lay3 = HillTransferFunctiontest(torch.tensor(n[2]), torch.tensor(K[2]))
# output1 = lay1(input)
# output2 = lay2(output1)
# output3 = lay3(output2)
# plt.scatter(input.detach().numpy(), output3.detach().numpy())
# output3.sum().backward()
