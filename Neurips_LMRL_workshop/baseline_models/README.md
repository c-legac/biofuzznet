Those are the notebooks that I've used mainly for the data analysis

- Bulking_data.ipynb: notebook containing my attempts at bulking the scRNAseq data and plots to see the results of the bulking

- Classification_clean.ipynb: contains attempts at classification models predicting cell function from CAR domain/TF expression as well as clean confusion matrix plots. Used to generate Figure 4 of the report

- Classify_from_node_expression.ipynb: other attempts at classification that were never successful/clean enough to go in an official report

- data_exploration.ipynb: plotting the expression/raw counts of nodes against that of their parents to see the amount of available signal

- Literature_GRN_sanity_check.ipynb: same as data_exploration except with nodes that are not from the PKN protein signaling but fromn the GRN, including scRNAseq from other datasets with the same genes

- MAGIC exploration.ipynb: notebook used to try and test MAGIC

- Poisson_to_remove_lines.ipynb: Code and tests for the function adding Poisson noise to the log1p-transformed gene expression in order to avoid having the weird lines

- Predict_new_combinations_from_domain_name.ipynb: this is the notebook which contains baseline models which try to predict the function of new domain combinations from the one-hot encoding of the CAR domain, as well as the probability distribution of cell function for those new domain combinations and a pvalue-like score to ascertain how close the predicted distribution is to the observed distribution

- Predict_TF_expression_from_all_network_without_GRN.ipynb: notebook containing regression models trying to predict TF expression from the expression of all other nodes in the network

- Raw_counts_exploration.ipynb: notebook containing the code used to evaluate the fraction of technical droout from the expression of household genes

- Stacked_barplot_from_AD.ipynb: notebook containing the code used to generate the cell function distibution barplots for the report (Figure 2)

- systematic_correlation_calculations.ipynb: notebook computing the correlation coefficients within the CAR dataset and the DREAM datase, used to generate Figure 7 and Table 2 of the report

- TF_expression_from_parent_noes.ipynb: notebook containing regression models trying to predict expression at transcription factors from that of their parent nodes. Used to generate Figure 5 of the report

- toy_network_data_exploration.ipynb: same as data_exploration and Literature_GRN_sanity_check on PDK1 and PCKt. Used to generate Figure 6b of the report