from sklearn.model_selection import ParameterSampler, ParameterGrid
import json
from biological_fuzzy_logic_networks.DREAM_analysis.scripts.train_network import (
    train_network,
)

assert ParameterGrid
assert ParameterSampler


def create_and_save_configs(sampled_params, base_config):

    config = base_config.copy()
    for key, value in sampled_params.items():
        config[key] = value

    config[
        "output_dir"
    ] = f"{base_config['output_dir']}{sampled_params['cell_line']}_{sampled_params['treatment']}_"

    config[
        "data_file"
    ] = f"/dccstor/ipc1/CAR/DREAMdata/{sampled_params['cell_line']}.csv"

    with open(f"{config['output_dir']}.config.json", "w") as fp:
        json.dump(config, fp)

    return config


def main(base_config, param_grid):

    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        create_and_save_configs(params, base_config)

        train_network(**params)


if __name__ == "__main__":

    param_grid = {
        "cell_line": ["Hs578T", "ZR75B", "HCC1428", "MDAMB436"],
        # "train_treatments": [],
        "test_treatment": ["iEGFR", "iMEK", "iPI3K", "iPKC"],
        # "learning_rate": [],
        # "n_epochs": [],
        # "batch_size": [],
    }

    base_config = {
        "pkn_sif": "/dccstor/ipc1/CAR/DREAMdata/DREAM_PKN_for_BFZ_input.sif",
        "network_class": "BioFuzzNet",
        "data_file": "",
        "output_dir": "/dccstor/ipc1/CAR/DREAM/Model/Test/cl_tr_test/",
        "time_point": 9,
        "non_marker_cols": ["treatment", "cell_line", "time", "cellID", "fileID"],
        "treatment_col_name": "treatment",
        "minmaxscale": True,
        "add_root_values": True,
        "input_value": 1,
        "train_treatments": None,
        "test_treatments": None,
        "learning_rate": 1e-3,
        "n_epochs": 20,
        "batch_size": 300,
    }

    main(base_config=base_config, param_grid=param_grid)
