from sklearn.model_selection import ParameterSampler, ParameterGrid
import json
import os

assert ParameterGrid
assert ParameterSampler


def create_and_save_configs(sampled_params, base_config, i):
    config = base_config.copy()
    for key, value in sampled_params.items():
        config[key] = value

    config["output_dir"] = f"{base_config['output_dir']}/{i}/"

    try:
        os.mkdir(config["output_dir"])
    except FileExistsError:
        print("Directory already exists")

    config[
        "data_file"
    ] = f"/dccstor/ipc1/CAR/DREAM/DREAMdata/Time_aligned_per_cell_line/{sampled_params['cell_lines']}.csv"
    config[
        "test_cell_lines"
    ] = f"/dccstor/ipc1/CAR/DREAM/DREAMdata/Time_aligned_per_cell_line/{sampled_params['cell_lines']}.csv"

    with open(f"{config['output_dir']}{i}_config.json", "w") as fp:
        json.dump(config, fp)

    with open(f"{base_config['output_dir']}/Configs/{i}_config.json", "w") as fp:
        json.dump(config, fp)


def main(base_config, param_grid):
    param_list = list(ParameterSampler(param_grid, 15))

    for i, params in enumerate(param_list):
        create_and_save_configs(params, base_config, i)


if __name__ == "__main__":
    param_grid = {
        # For testing in CD3 all cell lines not in SC1, SC2, and SC4 are used as test. For this the imTOR treatment is left out in ALL other cell lines as well.
        "cell_lines": [
            # "HCC1428", # Test SC2 iPKC
            "HCC70",
            # "AU565", # Test SC1
            "HCC202",  # Test SC2 iMEK
            "BT20",
            "MDAMB415",
            "HCC2157",
            "MCF10F",
            # "MDAMB436", # Test SC1
            "BT474",
            # "CAMA1", # Test SC4 (only full given)
            # "Hs578T", # Test SC2 iPKC
            "MCF7",
            "MDAMB175VII",
            "MCF10A",
            "UACC812",
            # "ZR751", # Test SC2 iMEK
            "ZR7530",
            # "CAL120", # Test SC4 (only full given)
            "HCC3153",
            # "EFM19", # Test SC1
            "CAL51",
            # "MACLS2", # Test SC1
            # "ZR75B", # Test SC4 (only full given)
            "CAL851",
            "EFM192A",
            "HBL100",
            # "LY2", # Test SC1
            "MPE600",
            "CAL148",
            "HDQP1",
            # "HCC1143", # Test SC4 (only full given
            "EVSAT",
            "MDAMB157",
            # "HCC2218", # Test SC1
            "HCC1599",
            "HCC2185",
            "DU4475",
            "OCUBM",
            # "184B5", # Test SC2 iMEK
            # "UACC3199", # Test SC2 iPI3K
            "184A1",
            "JIMT1",
            # "KPL1", # Test SC4 (only full given
            "MX1",
            "MFM223",
            # "BT483", # Test SC2 iEGFR
            "MDAMB453",
            "HCC1187",
            "HCC1419",
            "HCC1500",
            "HCC1395",
            # "HCC1806", # Test SC2 iPKC
            "HCC1937",
            "BT549",
            "HCC1954",
            # "MDAMB231", # Test SC2 iPI3K
            # "MCF12A", # Test SC2 iEGFR
            "MDAkb2",
            # "MDAMB468", # Test SC2 iEGFR
            "MDAMB134VI",
            # "SKBR3", # Test SC2 iPI3K
            "HCC38",
            "MDAMB361",
            "HCC1569",
            "UACC893",
            "T47D",
        ],
        # "train_treatments": [],
        # "valid_treatments": ["iEGFR", "iMEK", "iPI3K", "iPKC"],
        # "valid_cell_lines": [],
        # "learning_rate": [0.01, 0.005, 0.2],
        # "n_epochs": [10, 50, 100],
        "batch_size": [300, 1000, 10000],
    }

    base_config = {
        "pkn_sif": "/dccstor/ipc1/CAR/DREAM/DREAMdata/PKN_Alice.sif",
        "network_class": "DREAMBioFuzzNet",
        "data_file": "",
        "output_dir": "/dccstor/ipc1/CAR/DREAM/Model/Test/Loops/",
        "time_point": 9,
        "non_marker_cols": ["treatment", "cell_line", "time", "cellID", "fileID"],
        "treatment_col_name": "treatment",
        "minmaxscale": True,
        "add_root_values": True,
        "input_value": 1,
        "train_treatments": None,
        "valid_treatments": None,
        "train_cell_lines": None,
        "valid_cell_lines": None,
        "convergence_check": True,
        "inhibition_value": 10.0,
        "learning_rate": 1e-3,
        "n_epochs": 20,
        "batch_size": 300,
        "checkpoint_path": "/dccstor/ipc1/CAR/DREAM/Model/Test/Loops/",
        "experiment_name": "loops_test",
    }

    main(base_config=base_config, param_grid=param_grid)
