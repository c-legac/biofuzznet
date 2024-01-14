# Identifiability of incorrect edges

Used to generate the part of the thesis named: *Errors in the prior knowledge networks can be detected*

Contains:
    - generate_data.py: script used to generate the datasets
    - optimisation_script.py: script used to optimise the BioFuzzNet
    
   - dataset_MNfull_3600cells_seed_X: two datasets generated from the full manual CAR network with 2 different seeds (ie. 2 different transfer functions parameters of the BioFuzzNet). Datasets were generated from network Network.tsv
   - Network.tsv: network file used to build the BioFuzzNet generating the data
   
   
   - report_identifiability_AND_seed_X: two replicates of the optimisation of the network with incorrect edges linked to AND gates. Optimisation was done on a BioFuzzNet built from the file Network_spurious_edges_AND.tsv
   - Network_spurious_edges_AND.tsv: network file used to build the BioFuzzNet for test identifiability of spurious edges at AND gates
   
   
   - report_identifiability_OR_X: two replicates of the optimisation of the network with incorrect edges linked to OR gates. I forgot to write the seed: I assume that 1 is seed 2 and 2 is seed 2000 but I can't be sure. Optimisation was done on a BioFuzzNet built from the file Network_spurious_edges_OR.tsv
    - Network_spurious_edges_OR.tsv: network file used to build the BioFuzzNet for test identifiability of spurious edges at OR gates
   
   
