#!/bin/bash
source /opt/share/anaconda3-2019.03/x86_64/etc/profile.d/conda.sh
conda activate DREAM_analysis
echo $CONDA_DEFAULT_ENV

jbsub -q x86_12h -cores 8+2 -mem 200g python3 /u/adr/Code/biofuzznet/DREAM_analysis/scripts/fit_distributions.py -d "/dccstor/ipc1/CAR/Data/DREAM_data/all_cl_time_aligned.csv" -t 9.0 -dist weibull_min norm weibull_max beta invgauss uniform gamma expon lognorm pearson3 triang -o /dccstor/ipc1/CAR/Data/DREAM_data/Dist_fitting/t0.9_ -s "MinMaxScaler"

jbsub -q x86_12h -cores 8+2 -mem 200g python3 /u/adr/Code/biofuzznet/DREAM_analysis/scripts/fit_distributions.py -d "/dccstor/ipc1/CAR/Data/DREAM_data/all_cl_time_aligned.csv" -t 9.0 -dist weibull_min norm weibull_max beta invgauss uniform gamma expon lognorm pearson3 triang -o /dccstor/ipc1/CAR/Data/DREAM_data/Dist_fitting/t0.9_ -s "StandardScaler"