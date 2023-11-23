from biological_fuzzy_logic_networks.DREAM import DREAMBioFuzzNet
import torch
import numpy as np
import pandas as pd


def run_sim_and_baselines(
    pkn_path,
    train_size,
    test_size,
    inhibited_edge: tuple = ("mek12", "erk12"),
    k_inhibition: float = 5.0,
    divide_inhibition: float = 10.0,
    train_frac=0.7,
    BFN_training_params: dict = {
        "epochs": 100,
        "batch_size": 500,
        "learning_rate": 0.001,
        "tensors_to_cuda": True,
    },
):
    teacher_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )
    student_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )
    untrained_network = DREAMBioFuzzNet.DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(
        pkn_path
    )

    # INHIBITION INPUTS
    teacher_network.initialise_random_truth_and_output(train_size)
    no_inhibition = {k: torch.ones(train_size) for k in teacher_network.nodes}
    no_inhibition_test = {k: torch.ones(test_size) for k in teacher_network.nodes}
    perturb_inhibition = no_inhibition_test.copy()
    perturb_inhibition[inhibited_edge[0]] = torch.Tensor(
        [divide_inhibition] * test_size
    )

    # Generate training data without perturbation
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
    input_df = pd.DataFrame(input_data)

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

    # Generate test data with perturbation
    # Introduce perturbation
    original_param = teacher_network.edges[inhibited_edge]["layer"].K
    teacher_network.edges[inhibited_edge]["layer"].K = torch.nn.Parameter(
        torch.Tensor([k_inhibition])
    )

    # Generate test data with perturbation
    teacher_network.sequential_update(
        teacher_network.root_nodes, inhibition=no_inhibition
    )
    with torch.no_grad():
        perturb_data = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        perturb_input = {
            k: v
            for k, v in teacher_network.output_states.items()
            if k in teacher_network.root_nodes
        }

        perturb_true_df = pd.DataFrame({k: v.numpy() for k, v in perturb_data.items()})

    # Train student on unperturbed training data
    # Split train data in training and validation data
    train = train_true_df.sample(frac=train_frac)
    val = train_true_df.drop(train.index, axis=0)

    train_dict = {c: torch.Tensor(np.array(train[c])) for c in train.columns}
    val_dict = {c: torch.Tensor(np.array(val[c])) for c in val.columns}

    # Same input as teacher:
    train_input = input_df.iloc[train.index, :]
    val_input = input_df.drop(train.index, axis=0)
    train_input_dict = {
        c: torch.Tensor(np.array(train_input[c])) for c in train_input.columns
    }
    val_input_dict = {
        c: torch.Tensor(np.array(val_input[c])) for c in val_input.columns
    }

    # Data should have root nodes and non-root nodes
    val_dict.update(val_input_dict)
    train_dict.update(train_input_dict)

    # Inhibitor
    train_inhibitors = {c: torch.ones(len(train)) for c in train_dict.keys()}
    val_inhibitors = {c: torch.ones(len(val)) for c in val_dict.keys()}

    student_network.initialise_random_truth_and_output(train_size)
    losses, curr_best_val_loss, _ = student_network.conduct_optimisation(
        input=train_input_dict,
        ground_truth=train_dict,
        train_inhibitors=train_inhibitors,
        valid_ground_truth=val_dict,
        valid_input=val_input_dict,
        valid_inhibitors=val_inhibitors,
        **BFN_training_params,
    )

    # TEACHER network with division inhibition (K back to original), same inputs
    teacher_network.edges[("mek12", "erk12")]["layer"].K = original_param
    teacher_network.sequential_update(
        teacher_network.root_nodes, inhibition=perturb_inhibition
    )
    with torch.no_grad():
        true_with_i_data = {
            k: v.numpy()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        teach_div_perturb_with_i = pd.DataFrame(true_with_i_data)

    # TEACHER network with division inhibition (K back to original), random inputs
    teacher_network.initialise_random_truth_and_output(test_size)
    teacher_network.sequential_update(
        teacher_network.root_nodes, inhibition=perturb_inhibition
    )
    with torch.no_grad():
        true_wo_i_data = {
            k: v.numpy()
            for k, v in teacher_network.output_states.items()
            if k not in teacher_network.root_nodes
        }
        teach_div_perturb = pd.DataFrame(true_wo_i_data)

    # TEST student without perturbation, same inputs
    test_ground_truth = test_input.copy()
    test_ground_truth.update(test_data)
    student_network.initialise_random_truth_and_output(test_size)
    student_network.set_network_ground_truth(test_ground_truth)

    student_network.sequential_update(
        teacher_network.root_nodes, inhibition=no_inhibition_test
    )
    with torch.no_grad():
        test_output = {
            k: v
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        test_output_df = pd.DataFrame({k: v.numpy() for k, v in test_output.items()})

    # TEST student network without perturbation, random inputs
    student_network.initialise_random_truth_and_output(test_size)
    student_network.sequential_update(
        teacher_network.root_nodes, inhibition=no_inhibition_test
    )
    with torch.no_grad():
        test_random_output = {
            k: v
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        test_random_output_df = pd.DataFrame(
            {k: v.numpy() for k, v in test_random_output.items()}
        )

    # TEST student network with perturbation, same inputs
    perturb_ground_truth = perturb_input.copy()
    perturb_ground_truth.update(perturb_data)

    student_network.initialise_random_truth_and_output(test_size)
    student_network.set_network_ground_truth(perturb_ground_truth)

    student_network.sequential_update(
        teacher_network.root_nodes, inhibition=perturb_inhibition
    )
    with torch.no_grad():
        perturb_output = {
            k: v
            for k, v in student_network.output_states.items()
            if k not in student_network.root_nodes
        }
        perturb_output_df = pd.DataFrame(
            {k: v.numpy() for k, v in perturb_output.items()}
        )

    # TEST student network with perturbation, random inputs
    untrained_network.initialise_random_truth_and_output(test_size)
    untrained_network.sequential_update(
        untrained_network.root_nodes, inhibition=perturb_inhibition
    )
    with torch.no_grad():
        perturb_gen = {
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        perturb_gen_df = pd.DataFrame(perturb_gen)

    # UNTRAINED NETWORK on perturbation, same inputs
    untrained_network.initialise_random_truth_and_output(test_size)
    untrained_network.set_network_ground_truth(perturb_ground_truth)
    untrained_network.sequential_update(
        teacher_network.root_nodes, inhibition=perturb_inhibition
    )
    with torch.no_grad():
        ut_perturb_with_input = {
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        ut_perturb_with_input_df = pd.DataFrame(ut_perturb_with_input)

    # UNTRAINED NETWORK on perturbation, random inputs
    untrained_network.initialise_random_truth_and_output(test_size)
    untrained_network.sequential_update(
        untrained_network.root_nodes, inhibition=perturb_inhibition
    )
    with torch.no_grad():
        ut_perturb = {
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }

        ut_perturb_df = pd.DataFrame(ut_perturb)

    # UNTRAINED NETWORK without perturbation, same inputs
    untrained_network.initialise_random_truth_and_output(test_size)
    untrained_network.set_network_ground_truth(test_ground_truth)
    untrained_network.sequential_update(
        untrained_network.root_nodes, inhibition=no_inhibition_test
    )
    with torch.no_grad():
        gen_with_i_test = {
            k: v.numpy()
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
            k: v.numpy()
            for k, v in untrained_network.output_states.items()
            if k not in untrained_network.root_nodes
        }
        ut_test_df = pd.DataFrame(gen_test)

    # COMBINE all data
    test_true_df
    perturb_true_df
    teach_div_perturb_with_i  # teacher, perturb, division
    teach_div_perturb
    test_output_df  # student, no perturb same inputs
    test_random_output_df
    perturb_output_df  # Student perturb same inputs
    perturb_gen_df
    ut_perturb_with_input_df  # Untrained  same inputs
    ut_perturb_df
    ut_test_with_i_df  # Untrained no perturb, same inputs
    ut_test_df

    pertubed_pred_data = pd.concat(
        [
            teach_div_perturb_with_i,
            teach_div_perturb,
            perturb_output_df,
            perturb_gen_df,
            ut_perturb_with_input_df,
            ut_perturb_df,
        ],
        keys=[
            "teacher_division_same_input",
            "teacher_division_random_input",
            "student_division_same_input",
            "student_division_random_input",
            "untrained_division_same_input",
            "untrained_division_random_input",
        ],
    )

    unpertubed_pred_data = pd.concat(
        [
            test_output_df,
            test_random_output_df,
            ut_test_with_i_df,
            ut_test_df,
        ],
        keys=[
            "student_no_perturb_same_input",
            "student_no_perturb_random_input",
            "untrained_no_perturb_same_input",
            "untrained_no_perturb_random_input",
        ],
    )

    return losses, pertubed_pred_data, unpertubed_pred_data, losses


run_sim_and_baselines(pkn_path="/dccstor/ipc1/CAR/BFN/LiverDREAM_PKN.sif")
