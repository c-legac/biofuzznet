from biological_fuzzy_logic_networks.DREAM_analysis.utils import (
    create_bfz,
    prepare_cell_line_data,
    cl_data_to_input,
)
import pandas as pd
from typing import List, Union
import click
import json


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
    root_nodes: List[str] = ["EGF", "SERUM"],
    train_treatments: List[str] = None,
    valid_treatments: List[str] = None,
    train_cell_lines: List[str] = None,
    valid_cell_lines: List[str] = None,
    inhibition_value: Union[int, float] = 1.0,
    learning_rate: float = 1e-3,
    n_epochs: int = 20,
    batch_size: int = 300,
    checkpoint_path: str = None,
    convergence_check: bool = False,
    **extras,
):
    model = create_bfz(pkn_sif, network_class)
    cl_data = prepare_cell_line_data(
        data_file=data_file,
        time_point=time_point,
        non_marker_cols=non_marker_cols,
        treatment_col_name=treatment_col_name,
    )

    (
        train_data,
        valid_data,
        train_inhibitors,
        valid_inhibitors,
        train_input,
        valid_input,
        train,
        valid,
    ) = cl_data_to_input(
        data=cl_data,
        model=model,
        train_treatments=train_treatments,
        valid_treatments=valid_treatments,
        train_cell_lines=train_cell_lines,
        valid_cell_lines=valid_cell_lines,
        inhibition_value=inhibition_value,
        minmaxscale=minmaxscale,
        add_root_values=add_root_values,
        input_value=input_value,
        root_nodes=root_nodes,
    )

    loss, loop_states = model.conduct_optimisation(
        input=train_input,
        valid_input=valid_input,
        ground_truth=train_data,
        valid_ground_truth=valid_data,
        train_inhibitors=train_inhibitors,
        valid_inhibitors=valid_inhibitors,
        epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        convergence_check=convergence_check,
    )

    output_states = pd.DataFrame({k: v.numpy() for k, v in model.output_states.items()})

    loss.to_csv(f"{output_dir}loss.csv")
    output_states.to_csv(f"{output_dir}output_states.csv")
    train.to_csv(f"{output_dir}train_data.csv")
    valid.to_csv(f"{output_dir}valid_data.csv")
    pd.DataFrame(train_inhibitors).to_csv(f"{output_dir}train_inhibitors.csv")
    pd.DataFrame(valid_inhibitors).to_csv(f"{output_dir}valid_inhibitors.csv")

    if convergence_check:
        temp = {
            idx: {m: v.detach().numpy() for (m, v) in m.items()}
            for (idx, m) in loop_states.items()
        }
        loop_states_to_save = pd.concat(
            [pd.DataFrame(v) for k, v in temp.items()],
            keys=temp.keys(),
            names=["time", ""],
        ).reset_index("time", drop=False)
        loop_states_to_save.to_csv(f"{output_dir}loop_states.csv")


@click.command()
@click.argument("config_path")
def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    train_network(**config)


if __name__ == "__main__":
    main()
