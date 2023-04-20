from biological_fuzzy_logic_networks.DREAM.DREAMBioFuzzNet import (
    DREAMBioFuzzNet,
    DREAMBioMixNet,
)
from biological_fuzzy_logic_networks.utils import read_sif
from biological_fuzzy_logic_networks.biomixnet import BioMixNet
from biological_fuzzy_logic_networks.biofuzznet import BioFuzzNet

from biological_fuzzy_logic_networks.DREAM_analysis.utils import (
    data_to_nodes_mapping,
    inhibitor_mapping,
)

import pandas as pd
from typing import List
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import DoubleTensor
import click
import json


def create_bfz(pkn_sif: str, network_class: str):
    nodes, edges = read_sif(pkn_sif)
    if network_class.lower() == "dreambiofuzznet":
        model = DREAMBioFuzzNet(nodes, edges)

    elif network_class.lower() == "dreambiomixnet":
        model = DREAMBioMixNet(nodes, edges)

    elif network_class.lower() == "biomixnet":
        model = BioMixNet(nodes, edges)
    elif network_class.lower() == "biofuzznet":
        model = BioFuzzNet(nodes, edges)
    else:
        raise Exception(
            "network_class arguement not recognised, should be one of",
            "DREAMBioFuzzNet, DREAMBioMixNet, BioFuzzNet, BioMixNet",
        )

    return model


def prepare_cell_line_data(
    data_file: str,
    root_nodes: List[str] = ["EGF", "SERUM"],
    time_point: int = 9,
    non_marker_cols: List[str] = ["treatment", "cell_line", "time", "cellID", "fileID"],
    treatment_col_name: str = "treatment",
    minmaxscale: bool = True,
    add_root_values: bool = True,
    input_value: float = 1,
):
    cl_data = pd.read_csv(data_file)
    print(cl_data["cell_line"].unique())
    data_to_nodes_map = data_to_nodes_mapping()
    inhibitor_map = inhibitor_mapping()

    cl_data = cl_data[cl_data["time"] == time_point]

    cl_data.loc[:, "inhibitor"] = [
        inhibitor_map[treatment] for treatment in cl_data[treatment_col_name]
    ]

    cl_data = cl_data.rename(columns=data_to_nodes_map)

    markers = [c for c in cl_data.columns if c not in non_marker_cols + ["inhibitor"]]
    if minmaxscale:
        scaler = MinMaxScaler()
        cl_data[markers] = scaler.fit_transform(cl_data[markers])

    non_network_measurements = [
        m for m in markers if m not in data_to_nodes_map.values()
    ]
    cl_data = cl_data.drop(
        non_marker_cols + non_network_measurements,
        axis=1,
    )

    if add_root_values:
        cl_data.loc[:, root_nodes] = input_value
        markers = markers + root_nodes

    return cl_data, markers


def cl_data_to_input(
    data,
    markers,
    model,
    train_treatments: List[str] = None,
    test_treatments: List[str] = None,
):
    if train_treatments is None and test_treatments is None:
        train, test = train_test_split(data)

    elif train_treatments is not None and test_treatments is not None:
        if not isinstance(train_treatments, List):
            train_treatments = [train_treatments]
        if not isinstance(test_treatments, List):
            test_treatments = [test_treatments]

        if not len(set(train_treatments).intersection(set(test_treatments))) == 0:
            raise Exception("Given train and test treatments overlap")
        else:
            inhibitor_map = inhibitor_mapping()
            train_inhibitors = [
                inhibitor_map[treatment] for treatment in train_treatments
            ]
            test_inhibitors = [
                inhibitor_map[treatment] for treatment in test_treatments
            ]

            train = data.loc[data["inhibitor"].isin(train_inhibitors)]
            test = data.loc[data["inhibitor"].isin(test_inhibitors)]

    elif train_treatments is not None:
        if not isinstance(train_treatments, List):
            train_treatments = [train_treatments]

        inhibitor_map = inhibitor_mapping()
        train_inhibitors = [inhibitor_map[treatment] for treatment in train_treatments]
        test_inhibitors = [
            inhibitor
            for inhibitor in data["inhibitor"].unique()
            if inhibitor not in train_inhibitors
        ]
        train = data.loc[data["inhibitor"].isin(train_inhibitors)]
        test = data.loc[data["inhibitor"].isin(test_inhibitors)]
    elif test_treatments is not None:
        print("Splitting based on test treatment")
        if not isinstance(test_treatments, List):
            test_treatments = [test_treatments]
        print(test_treatments)

        inhibitor_map = inhibitor_mapping()
        test_inhibitors = [inhibitor_map[treatment] for treatment in test_treatments]
        train_inhibitors = [
            inhibitor
            for inhibitor in data["inhibitor"].unique()
            if inhibitor not in test_inhibitors
        ]
        train = data.loc[data["inhibitor"].isin(train_inhibitors)]
        test = data.loc[data["inhibitor"].isin(test_inhibitors)]

    # print(train_inhibitors)
    # print(test_inhibitors)
    print(train["inhibitor"].unique())
    print(test["inhibitor"].unique())

    train_dict = train.to_dict("list")
    test_dict = test.to_dict("list")
    train_data = {k: DoubleTensor(v) for k, v in train_dict.items() if k in markers}
    test_data = {k: DoubleTensor(v) for k, v in test_dict.items() if k in markers}

    train_inhibitors = {
        m1: DoubleTensor([10.0 if m == m1 else 1.0 for m in train_dict["inhibitor"]])
        for m1 in model.nodes()
    }
    test_inhibitors = {
        m1: DoubleTensor([10.0 if m == m1 else 1.0 for m in test_dict["inhibitor"]])
        for m1 in model.nodes()
    }

    train_input = {node: train_data[node] for node in model.root_nodes}
    test_input = {node: test_data[node] for node in model.root_nodes}

    return (
        train_data,
        test_data,
        train_inhibitors,
        test_inhibitors,
        train_input,
        test_input,
    )


def train_network(
    pkn_sif: str,
    network_class: str,
    data_file: str,
    output_dir: str,
    time_point: int = 9,
    non_marker_cols: List[str] = ["treatment", "cell_line", "time", "cellID", "fileID"],
    treatment_col_name: str = "treatment",
    minmaxscale: bool = True,
    add_root_values: bool = True,
    input_value: float = 1,
    train_treatments: List[str] = None,
    test_treatments: List[str] = None,
    learning_rate: float = 1e-3,
    n_epochs: int = 20,
    batch_size: int = 300,
    **extras,
):

    model = create_bfz(pkn_sif, network_class)
    cl_data, markers = prepare_cell_line_data(
        data_file=data_file,
        root_nodes=model.root_nodes,
        time_point=time_point,
        non_marker_cols=non_marker_cols,
        treatment_col_name=treatment_col_name,
        minmaxscale=minmaxscale,
        add_root_values=add_root_values,
        input_value=input_value,
    )

    (
        train_data,
        test_data,
        train_inhibitors,
        test_inhibitors,
        train_input,
        test_input,
    ) = cl_data_to_input(
        data=cl_data,
        model=model,
        markers=markers,
        train_treatments=train_treatments,
        test_treatments=test_treatments,
    )

    loss = model.conduct_optimisation(
        input=train_input,
        test_input=test_input,
        ground_truth=train_data,
        test_ground_truth=test_data,
        train_inhibitors=train_inhibitors,
        test_inhibitors=test_inhibitors,
        epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    output_states = pd.DataFrame({k: v.numpy() for k, v in model.output_states.items()})

    loss.to_csv(f"{output_dir}loss.csv")
    output_states.to_csv(f"{output_dir}output_states.csv")
    train_data.to_csv(f"{output_dir}train_data.csv")
    test_data.to_csv(f"{output_dir}test_data.csv")


@click.command()
@click.argument("config_path")
def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    train_network(**config)


if __name__ == "__main__":
    main()
