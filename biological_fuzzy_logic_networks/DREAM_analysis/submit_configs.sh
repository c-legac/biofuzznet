#!/bin/bash
source /opt/share/anaconda3-2019.03/x86_64/etc/profile.d/conda.sh
conda activate biofuzznet
echo $CONDA_DEFAULT_ENV

config_folder=/dccstor/ipc1/CAR/DREAM/Model/Test/cl_tr_test/

for file in $config_folder*.json
    do 
        jbsub -q x86_12h -mem 100g -cores 8+2 python /u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/DREAM_analysis/train_network.py "$file"
    done