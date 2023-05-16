#!/bin/bash
source /opt/share/anaconda3-2019.03/x86_64/etc/profile.d/conda.sh
conda activate biofuzznet2
echo $CONDA_DEFAULT_ENV

config_folder=/dccstor/ipc1/CAR/DREAM/Model/Test/OneCellLineOneTreatment_MEK_FAK_ERK/Configs/

# for i in 66 392 291 226

for file in $config_folder*.json
    do 
        # echo ${config_folder}${i}_config.json
        jbsub -q x86_6h -mem 100g -cores 2+2 python /u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/DREAM_analysis/train_network_per_CL_treatment.py $file
    done