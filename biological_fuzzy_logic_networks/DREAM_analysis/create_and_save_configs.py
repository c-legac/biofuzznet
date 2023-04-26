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

    config["data_file"] = [
        f"/dccstor/ipc1/CAR/DREAM/DREAMdata/{cl}.csv"
        for cl in sampled_params["cell_lines"]
    ]
    config["learning_rate"] = sampled_params["learning_rate"]
    config["n_epochs"] = sampled_params["n_epochs"]
    config["batch_size"] = sampled_params["batch_size"]

    with open(f"{config['output_dir']}{i}_config.json", "w") as fp:
        json.dump(config, fp)

    with open(f"{base_config['output_dir']}/Configs/{i}_config.json", "w") as fp:
        json.dump(config, fp)


def main(base_config, param_grid):
    param_list = list(ParameterGrid(param_grid))

    for i, params in enumerate(param_list):
        create_and_save_configs(params, base_config, i)


if __name__ == "__main__":
    param_grid = {
        "cell_lines": [
            [
                "HCC1428",
                "HCC70",
                # "AU565", # Test
                "HCC202",
                "BT20",
                "MDAMB415",
                "HCC2157",
                "MCF10F",
                # "MDAMB436",
                "BT474",
                "CAMA1",
                "Hs578T",
                "MCF7",
                "MDAMB175VII",
                "MCF10A",
                "UACC812",
                "ZR751",
                "ZR7530",
                "CAL120",
                "HCC3153",
                # "EFM19",
                "CAL51",
                # "MACLS2",
                "ZR75B",
                "CAL851",
                "EFM192A",
                "HBL100",
                # "LY2",
                "MPE600",
                "CAL148",
                "HDQP1",
                "HCC1143",
                "EVSAT",
                "MDAMB157",
                # "HCC2218",
                "HCC1599",
                "HCC2185",
                "DU4475",
                "OCUBM",
                "184B5",
                "UACC3199",
                "184A1",
                "JIMT1",
                "KPL1",
                "MX1",
                "MFM223",
                "BT483",
                "MDAMB453",
                "HCC1187",
                "HCC1419",
                "HCC1500",
                "HCC1395",
                "HCC1806",
                "HCC1937",
                "BT549",
                "HCC1954",
                "MDAMB231",
                "MCF12A",
                "MDAkb2",
                "MDAMB468",
                "MDAMB134VI",
                "SKBR3",
                "HCC38",
                "MDAMB361",
                "HCC1569",
                "UACC893",
                "T47D",
            ]
        ],
        # "train_treatments": [],
        # "valid_treatments": ["iEGFR", "iMEK", "iPI3K", "iPKC"],
        "valid_cell_lines": [
            [
                "MCF12A",
                "HCC38",
                "T47D",
                "UACC812",
                "EFM192A",
                "MDAMB468",
                "JIMT1",
            ]
        ],
        "learning_rate": [0.01, 0.005, 0.2],
        "n_epochs": [10, 50, 100],
        "batch_size": [10, 300, 1000],
    }

    base_config = {
        "pkn_sif": "/dccstor/ipc1/CAR/DREAM/DREAMdata/MEK_FAK_ERK.sif",
        "network_class": "DREAMBioFuzzNet",
        "data_file": "",
        "output_dir": "/dccstor/ipc1/CAR/DREAM/Model/Test/MEK_FAK_ERK/",
        "time_point": 9,
        "non_marker_cols": ["treatment", "cell_line", "time", "cellID", "fileID"],
        "treatment_col_name": "treatment",
        "minmaxscale": True,
        "add_root_values": False,
        "input_value": 1,
        "train_treatments": None,
        "valid_treatments": None,
        "train_cell_lines": None,
        "valid_cell_lines": None,
        "inhibition_value": 1.0,
        "learning_rate": 1e-3,
        "n_epochs": 20,
        "batch_size": 300,
        "checkpoint_path": "/dccstor/ipc1/CAR/DREAM/Model/Test/",
    }

    main(base_config=base_config, param_grid=param_grid)
