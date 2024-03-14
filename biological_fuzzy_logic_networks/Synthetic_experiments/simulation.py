from biological_fuzzy_logic_networks.DREAM import DREAMBioFuzzNet

import torch
import pandas as pd
import json
import click


def simulate_data(
    pkn_path,
    train_size,
    test_size,
    BFN_training_params: dict = {
        "epochs": 100,
        "batch_size": 500,
        "learning_rate": 0.001,
        "tensors_to_cuda": True,
    },
    **extras,
):
    teacher_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )

    # INHIBITION INPUTS
    no_inhibition = {k: torch.ones(train_size) for k in teacher_network.nodes}
    no_inhibition_test = {k: torch.ones(test_size) for k in teacher_network.nodes}

    # Generate training data without perturbation
    teacher_network.initialise_random_truth_and_output(train_size)
    teacher_network.sequential_update(
        teacher_network.root_nodes, inhibition=no_inhibition
    )
    with torch.no_grad():
        true_unperturbed_data = {
            k: v.numpy()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        input_data = pd.DataFrame(
            {
                k: v.numpy()
                for k, v in teacher_network.output_states.items()
                if k in teacher_network.root_nodes
            }
        )

    train_true_df = pd.DataFrame(true_unperturbed_data)
    train_input_df = pd.DataFrame(input_data)

    # Generate test data without perturbation
    teacher_network.initialise_random_truth_and_output(test_size)
    teacher_network.sequential_update(
        teacher_network.root_nodes, inhibition=no_inhibition_test
    )
    with torch.no_grad():
        test_data = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        test_input = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k in teacher_network.root_nodes
        }

        test_true_df = pd.DataFrame({k: v.numpy() for k, v in test_data.items()})
        test_input_df = pd.DataFrame({k: v.numpy() for k, v in test_input.items()})

    return train_true_df, train_input_df, test_true_df, test_input_df, teacher_network


@click.command()
@click.argument("config_path")
def main(config_path):
    print("Started")
    with open(config_path) as f:
        config = json.load(f)
    f.close()
    print("Loaded config")

    train_true_df, train_input_df, test_true_df, test_input_df, model = simulate_data(
        **config
    )
    print("Created data")

    train_true_df.to_csv(f"{config['out_dir']}train_true_df.csv", index=False)
    train_input_df.to_csv(f"{config['out_dir']}train_input_df.csv", index=False)
    test_true_df.to_csv(f"{config['out_dir']}test_true_df.csv", index=False)
    test_input_df.to_csv(f"{config['out_dir']}test_input_df.csv", index=False)

    torch.save(
        {"model_state_dict": model}, f"{config['out_dir']}model_for_simulation.pt"
    )

    with open(f"{config['out_dir']}/config.json", "w") as f:
        json.dump(config, f)

    print("Saved data and model")


if __name__ == "__main__":
    main()
