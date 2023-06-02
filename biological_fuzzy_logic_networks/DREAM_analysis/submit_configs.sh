#!/bin/bash
source /opt/share/anaconda3-2019.03/x86_64/etc/profile.d/conda.sh
conda activate biofuzznet2
echo $CONDA_DEFAULT_ENV

config_folder=/dccstor/ipc1/CAR/DREAM/Model/Test/Subnetwork/Configs/

for file in $config_folder*.json
    do 
        jbsub -q x86_24h -mem 100g -cores 8+2 python /u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/DREAM_analysis/train_network.py "$file"
    done