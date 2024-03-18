#!/bin/bash
source /opt/share/anaconda3-2019.03/x86_64/etc/profile.d/conda.sh
conda activate biofuzznet2
echo $CONDA_DEFAULT_ENV

config_folder=/dccstor/ipc1/CAR/BFN/Model/Noise/Configs/

for file in $config_folder*.json
    do 
        # echo ${file}
        jbsub -q x86_6h -cores 1+1 -mem 100g python /u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/training_noise.py $file
    done