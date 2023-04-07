import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    Normalizer,
    RobustScaler,
    MaxAbsScaler,
)
import scipy
import argparse


def parse_argmuments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        metavar="exprsseion_data_file_path",
        type=str,
        help="path to full expression file (.csv)",
    )
    parser.add_argument(
        "-t",
        metavar="selected_timepoint",
        help="run analysis for these timepoints only",
        required=False,
        nargs="*",
    )
    parser.add_argument(
        "-c",
        metavar="cell_lines",
        required=False,
        help="run analysis on these cell lines only",
        nargs="*",
    )
    parser.add_argument(
        "-tr",
        metavar="treatments",
        required=False,
        help="run analysis on these treatments only",
        nargs="*",
    )
    parser.add_argument(
        "-dist",
        metavar="distributions",
        help="names of scipy distributions to test",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        metavar="output_folder",
        help="output folder where to save the results",
    )

    parser.add_argument(
        "-s",
        metavar="scaler",
        help="which scaler to use for the data",
        choices=[
            "MaxAbsScaler",
            "MinMaxScaler",
            "Normalizer",
            "RobustScaler",
            "StandardScaler",
        ],
    )
    args = parser.parse_args()

    return args


def get_scaler(scaler_name):

    if scaler_name == "MaxAbsScaler":
        scaler = MaxAbsScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_name == "Normalizer":
        scaler = Normalizer()
    elif scaler_name == "RobustScaler":
        scaler = RobustScaler()
    elif scaler_name == "StandardScaler":
        scaler = StandardScaler()

    return scaler


def read_args(args):

    data_file = args.d
    output_path = args.o
    cell_lines = args.c
    treatments = args.tr
    times = [float(t) for t in args.t]
    dist_names = args.dist
    scaler_name = args.s

    data = pd.read_csv(data_file)

    return data, output_path, dist_names, cell_lines, times, treatments, scaler_name


def fit_distribution(df, column, dist_names, n_bins=11):
    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    y_std = list(df[column])
    size = len(df[column])

    chi_square_statistics = []
    # 11 bins
    bins = pd.cut(df[column], n_bins).cat.categories
    cum_observed_frequency = list((pd.cut(df[column], n_bins)).value_counts())
    percentile_cutoffs = [bins[0].left] + [iv.right for iv in bins]

    # Loop through candidate distributions

    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)

        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin_ in range(n_bins):
            expected_cdf_area = cdf_fitted[bin_ + 1] - cdf_fitted[bin_]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)

        ss = sum(
            ((cum_expected_frequency - cum_observed_frequency) ** 2)
            / cum_observed_frequency
        )
        chi_square_statistics.append(ss)

    return chi_square_statistics


def fit_cell_lines_to_distributions(data, dist_names, scaler):

    ident_cols = ["treatment", "cell_line", "time", "cellID", "fileID", "time_course"]
    markers = [c for c in data.columns if c not in ident_cols]
    sel_markers = [m for m in markers if m not in ["p.HER2", "p.PLCg2"]]

    all_stats = []
    all_cl = []
    all_treatments = []
    all_markers = []
    all_dists = []

    for cell_line in data["cell_line"].unique():
        cl_data = data[data["cell_line"] == cell_line]
        for treatment in cl_data["treatment"].unique():
            print(cell_line, treatment)
            sel_data = cl_data[cl_data["treatment"] == treatment]

            minmax_data = sel_data.copy()
            minmax_data[sel_markers] = scaler.fit_transform(sel_data[sel_markers])

            for marker in sel_markers:
                stats = fit_distribution(
                    minmax_data, column=marker, dist_names=dist_names
                )
                all_stats.append(stats)
                all_cl.append([cell_line] * len(dist_names))
                all_treatments.append([treatment] * len(dist_names))
                all_markers.append([marker] * len(dist_names))
                all_dists.append(dist_names)

    res = pd.DataFrame(
        {
            "chi_squared": [ss for ss_list in all_stats for ss in ss_list],
            "cell_line": [cl for cl_list in all_cl for cl in cl_list],
            "treatment": [t for t_list in all_treatments for t in t_list],
            "marker": [m for m_list in all_markers for m in m_list],
            "distribution": [d for d_list in all_dists for d in d_list],
        }
    )

    return res


def main(data, output_path, dist_names, cell_lines, times, treatments, scaler_name):

    if times is not None:
        data = data[data["time"].isin(times)]
    if cell_lines is not None:
        data = data[data["cell_line"].isin(cell_lines)]
    if treatments is not None:
        data = data[data["treatment"].isin(treatments)]

    print(data.head())

    scaler = get_scaler(scaler_name)

    dist_test_result = fit_cell_lines_to_distributions(data, dist_names, scaler)

    dist_test_result.to_csv(f"{output_path}{scaler_name}_dist_fitted.csv")


if __name__ == "__main__":
    args = parse_argmuments()
    (
        data,
        output_path,
        dist_names,
        cell_lines,
        times,
        treatments,
        scaler_name,
    ) = read_args(args)

    main(data, output_path, dist_names, cell_lines, times, treatments, scaler_name)
