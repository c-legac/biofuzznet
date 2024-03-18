from biological_fuzzy_logic_networks.DREAM import DREAMBioFuzzNet

import torch
import numpy as np
import pandas as pd
import json
import click
import os


def load_and_prepare_data(
    data_dir,
    noise_sd,
    add_noise_to_input_train,
    add_noise_to_y_train,
    add_noise_to_input_test,
    add_noise_to_y_test,
):
    train_true_df = pd.read_csv(f"{data_dir}/train_true_df.csv")
    train_input_df = pd.read_csv(f"{data_dir}/train_input_df.csv")
    test_true_df = pd.read_csv(f"{data_dir}/test_true_df.csv")
    test_input_df = pd.read_csv(f"{data_dir}/test_input_df.csv")

    if add_noise_to_y_train:
        # Add noise to training data (and test data?)
        train_input_noise = np.random.normal(0, noise_sd, train_input_df.shape)
        train_input_df = train_input_df + train_input_noise
        train_input_df[train_input_df <= 0] = 1e-9

    if add_noise_to_y_train:
        # Add noise to training data (and test data?)
        train_noise = np.random.normal(0, noise_sd, train_true_df.shape)
        train_true_df = train_true_df + train_noise
        train_true_df[train_true_df <= 0] = 1e-9

    if add_noise_to_input_test:
        # Add noise to test data
        input_noise = np.random.normal(0, noise_sd, test_input_df.shape)
        test_input_df = test_input_df + input_noise
        test_input_df[test_input_df <= 0] = 1e-9

    if add_noise_to_y_test:
        test_noise = np.random.normal(0, noise_sd, test_true_df.shape)
        test_true_df = test_true_df + test_noise
        test_true_df[test_true_df <= 0] = 1e-9

    return train_true_df, train_input_df, test_true_df, test_input_df


def run_train_with_noise(
    pkn_path,
    data_dir,
    train_frac: float = 0.7,
    noise_sd: int = 1,
    add_noise_to_input_train: bool = False,
    add_noise_to_y_train: bool = False,
    add_noise_to_input_test: bool = True,
    add_noise_to_y_test: bool = False,
    BFN_training_params: dict = {
        "epochs": 100,
        "batch_size": 500,
        "learning_rate": 0.001,
        "tensors_to_cuda": True,
    },
    **extras,
):

    student_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )
    untrained_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )

    # Get data with/without noise
    train_true_df, train_input_df, test_true_df, test_input_df = load_and_prepare_data(
        data_dir=data_dir,
        add_noise_to_input_train=add_noise_to_input_train,
        add_noise_to_y_train=add_noise_to_y_train,
        add_noise_to_input_test=add_noise_to_input_test,
        add_noise_to_y_test=add_noise_to_y_test,
        noise_sd=noise_sd,
    )
    test_size = len(test_true_df)

    # Train student on unperturbed training data
    # Split train data in training and validation data
    train = train_true_df.sample(frac=train_frac)
    val = train_true_df.drop(train.index, axis=0)
    train_size = len(train)
    val_size = len(val)

    train_dict = {c: torch.Tensor(np.array(train[c])) for c in train.columns}
    val_dict = {c: torch.Tensor(np.array(val[c])) for c in val.columns}

    # Same input as teacher:
    train_input = train_input_df.iloc[train.index, :]
    val_input = train_input_df.drop(train.index, axis=0)

    train_input_dict = {
        c: torch.Tensor(np.array(train_input[c])) for c in train_input.columns
    }
    val_input_dict = {
        c: torch.Tensor(np.array(val_input[c])) for c in val_input.columns
    }

    # Data should have root nodes and non-root nodes
    val_dict.update(val_input_dict)
    train_dict.update(train_input_dict)

    # Inhibitors
    train_inhibitors = {c: torch.ones(train_size) for c in train_dict.keys()}
    val_inhibitors = {c: torch.ones(val_size) for c in val_dict.keys()}

    student_network.initialise_random_truth_and_output(
        train_size, to_cuda=BFN_training_params["tensors_to_cuda"]
    )
    losses, curr_best_val_loss, _ = student_network.conduct_optimisation(
        input=train_input_dict,
        ground_truth=train_dict,
        train_inhibitors=train_inhibitors,
        valid_ground_truth=val_dict,
        valid_input=val_input_dict,
        valid_inhibitors=val_inhibitors,
        **BFN_training_params,
    )

    # TEST student without perturbation, same inputs
    test_ground_truth = {
        c: torch.Tensor(np.array(test_true_df[c])) for c in test_true_df.columns
    }
    test_input_dict = {
        c: torch.Tensor(np.array(test_input_df[c])) for c in test_input_df.columns
    }
    test_ground_truth.update(test_input_dict)
    no_inhibition_test = {k: torch.ones(test_size) for k in student_network.nodes}
    student_network.initialise_random_truth_and_output(
        test_size, to_cuda=BFN_training_params["tensors_to_cuda"]
    )
    student_network.set_network_ground_truth(
        test_ground_truth, to_cuda=BFN_training_params["tensors_to_cuda"]
    )

    student_network.sequential_update(
        student_network.root_nodes,
        inhibition=no_inhibition_test,
        to_cuda=BFN_training_params["tensors_to_cuda"],
    )
    with torch.no_grad():
        test_output = {
            k: v.cpu()
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        test_output_df = pd.DataFrame({k: v.numpy() for k, v in test_output.items()})

    # TEST student network without perturbation, random inputs
    student_network.initialise_random_truth_and_output(
        test_size, to_cuda=BFN_training_params["tensors_to_cuda"]
    )
    student_network.sequential_update(
        student_network.root_nodes,
        inhibition=no_inhibition_test,
        to_cuda=BFN_training_params["tensors_to_cuda"],
    )
    with torch.no_grad():
        test_random_output = {
            k: v.cpu()
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        test_random_output_df = pd.DataFrame(
            {k: v.numpy() for k, v in test_random_output.items()}
        )

    # UNTRAINED NETWORK without perturbation, same inputs
    untrained_network.initialise_random_truth_and_output(test_size)
    untrained_network.set_network_ground_truth(test_ground_truth)
    untrained_network.sequential_update(
        untrained_network.root_nodes, inhibition=no_inhibition_test
    )
    with torch.no_grad():
        gen_with_i_test = {
            k: v.cpu().numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        ut_test_with_i_df = pd.DataFrame(gen_with_i_test)

    # UNTRAINED NETWORK without perturbation, random inputs
    untrained_network.initialise_random_truth_and_output(test_size)
    untrained_network.sequential_update(
        untrained_network.root_nodes, inhibition=no_inhibition_test
    )
    with torch.no_grad():
        gen_test = {
            k: v.cpu().numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        ut_test_df = pd.DataFrame(gen_test)

    unpertubed_pred_data = pd.concat(
        [
            test_true_df,
            test_output_df,
            test_random_output_df,
            ut_test_with_i_df,
            ut_test_df,
        ],
        keys=[
            "teacher_true",
            "student_same_input",
            "student_random_input",
            "untrained_same_input",
            "untrained_random_input",
        ],
    )

    return (
        losses,
        unpertubed_pred_data,
        student_network.get_checkpoint(),
    )


@click.command()
@click.argument("config_path")
def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    out_dir = f"{config['out_dir']}/Error_sd_{config['noise_sd']}/"
    subdir = []
    if config["add_noise_to_input_train"]:
        subdir.append("tri")
    if config["add_noise_to_y_train"]:
        subdir.append("try")
    if config["add_noise_to_input_test"]:
        subdir.append("tei")
    if config["add_noise_to_y_test"]:
        subdir.append("tey")
    out_dir = out_dir + "_".join(subdir) + "/"

    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    base_data_dir = config["data_dir"]

    for i in range(5):

        config["data_dir"] = base_data_dir + str((i + 1)) + "/"
        with open(f"{config['data_dir']}/config.json") as f:
            sim_config = json.load(f)

        config["pkn_path"] = sim_config["pkn_path"]

        losses, unpertubed_data, student = run_train_with_noise(**config)

        losses.to_csv(f"{out_dir}{i+1}_losses.csv")
        unpertubed_data.to_csv(f"{out_dir}{i+1}_unperturbed.csv")

        torch.save({"model_state_dict": student}, f"{out_dir}{i+1}_student.pt")

        with open(f"{out_dir}{i+1}_config.json", "w") as f:
            json.dump(config, f)


if __name__ == "__main__":
    main()
