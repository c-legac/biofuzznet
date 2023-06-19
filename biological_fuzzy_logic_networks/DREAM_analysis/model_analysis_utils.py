from math import sqrt
import torch
import json
import pandas as pd
from biological_fuzzy_logic_networks.DREAM_analysis.utils import (
    create_bfz,
    prepare_cell_line_data,
    cl_data_to_input,
)


def mean_RMSE(
    ground_truth,
    predictions,
    markers=["p.ERK", "p.Akt.Ser473.", "p.S6", "p.HER2", "p.PLCg2"],
    non_marker_cols=[
        "cellID",
        "cell_line",
        "fileID",
        "glob_cellID",
        "time",
        "treatment",
    ],
    group_cols=["cell_line", "treatment", "time"],
):
    truth_non_marker_cols = set(ground_truth.columns).intersection(set(non_marker_cols))
    pred_non_marker_cols = set(predictions.columns).intersection(set(non_marker_cols))
    merge_cols = list(truth_non_marker_cols.intersection(pred_non_marker_cols))

    assert len(set(group_cols).intersection(merge_cols)) == len(group_cols)

    sel_truth = ground_truth[list(merge_cols) + markers]
    sel_pred = predictions[list(merge_cols) + markers]

    sel_truth = sel_truth.melt(
        id_vars=merge_cols, value_vars=markers, var_name="marker", value_name="true"
    )
    sel_pred = sel_pred.melt(
        id_vars=merge_cols, value_vars=markers, var_name="marker", value_name="pred"
    )

    assert len(sel_truth) == len(sel_pred)

    both = sel_truth.merge(sel_pred, on=merge_cols + ["marker"]).reset_index(drop=True)

    RMSE = both.groupby(["cell_line", "time", "treatment", "marker"]).apply(
        lambda df: sqrt(sum((df["true"] - df["pred"]) ** 2 / len(df)))
    )

    return RMSE.mean()


def get_scaler(run_folder, data_folder):
    print(run_folder)
    ckpt = torch.load(f"{run_folder}model.pt")
    with open(f"{run_folder}{run_folder[47:-1]}_config.json") as f:
        config = json.load(f)
    print(config["valid_cell_lines"])
    model = create_bfz(
        f"{data_folder}DREAMdata/MEK_FAK_ERK.sif", config["network_class"]
    )
    model.load_from_checkpoint(ckpt["model_state_dict"])

    cl_data = prepare_cell_line_data(**config)

    (
        train_data,
        valid_data,
        train_inhibitors,
        valid_inhibitors,
        train_input,
        valid_input,
        train,
        valid,
        scaler,
    ) = cl_data_to_input(
        data=cl_data,
        model=model,
        train_treatments=config["train_treatments"],
        valid_treatments=config["valid_treatments"],
        train_cell_lines=config["train_cell_lines"],
        valid_cell_lines=config["valid_cell_lines"],
        inhibition_value=config["inhibition_value"],
        minmaxscale=True,
        add_root_values=config["add_root_values"],
        input_value=1,
    )

    return scaler


def get_test_data_formatted(run_folder, data_folder):
    scaler = get_scaler(run_folder, data_folder)

    test_output = pd.read_csv(
        f"{run_folder}test_output_states.csv", index_col=0
    ).reset_index(drop=True)
    test_data = pd.read_csv(f"{run_folder}test_data.csv", index_col=0).reset_index(
        drop=True
    )

    test_unscaled = pd.DataFrame(
        scaler.inverse_transform(test_output[["ERK12", "FAK", "MEK12"]]),
        columns=test_output.columns,
        index=test_output.index,
    )
    test_output = test_output.rename(columns={"ERK12": "p.ERK"})
    test_output[["treatment", "cell_line", "time", "cellID", "fileID"]] = test_data[
        ["treatment", "cell_line", "time", "cellID", "fileID"]
    ]
    test_unscaled = test_unscaled.rename(columns={"ERK12": "p.ERK"})
    test_unscaled[["treatment", "cell_line", "time", "cellID", "fileID"]] = test_data[
        ["treatment", "cell_line", "time", "cellID", "fileID"]
    ]

    return test_output, test_unscaled
