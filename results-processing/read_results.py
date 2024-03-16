import os
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_results(results_file: Path) -> pd.DataFrame:
    """
    Reads a results file and returns a Pandas DataFrame.

    Parameters:
        results_file (Path): The path to the results file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the results from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    with open(results_file, encoding="utf-8") as f:
        data = []
        for l in f.readlines()[1:]:
            line = re.split(r"\t+", l)
            # Remove the 'K' column for the general model
            if len(line) == 10:
                line = line[:4] + line[5:]
            for i, v in enumerate(line):
                line[i] = v.strip()
            data.append(line)

    df = pd.DataFrame(
        data,
        columns=[
            "instance",
            "n",
            "Gamma",
            "Delta",
            "status",
            "objbound",
            "objval",
            "mipgap",
            "runtime",
        ],
    )

    df = df.astype(
        {
            "instance": int,
            "n": int,
            "Gamma": int,
            "Delta": int,
            "status": int,
            "objbound": float,
            "objval": float,
            "mipgap": float,
            "runtime": float,
        }
    )
    return df


def get_performance_profiles(
    results: Dict[str, pd.DataFrame], n: List[int]
) -> Dict[str, dict]:
    """
    Compute performance profiles for a set of methods based on their runtime data.

    Parameters:
        results (Dict[str, pd.DataFrame]): A dictionary of method names as keys and
            their corresponding runtime data as values in pandas DataFrames.
        n (List[int]): Instance sizes to include in plots.

    Returns:
        Dict[str, dict]: A dictionary of method names as keys and their corresponding
            performance profiles as values in dictionaries.

    """

    # Filter results for each method by values in n
    filtered_results = {}
    for m, data in results.items():
        filtered_results[m] = data[data["n"].isin(n)]

    # Format runtimes
    for m, data in filtered_results.items():
        data.loc[data["runtime"] == 0.00, "runtime"] = 0.01
        data.loc[data["runtime"] > 600, "runtime"] = 600

    # Compute performance ratios
    t = {m: data["runtime"].tolist() for m, data in filtered_results.items()}
    min_t = [min(t) for t in zip(*t.values())]
    p = {
        m: [700 if t[m][i] == 600 else t[m][i] / min_t[i] for i, _ in enumerate(t[m])]
        for m in filtered_results
    }

    # Get performance profiles
    I = len(p[m])
    data_points = 1000
    performance_profiles = {m: {} for m in filtered_results}
    for idx in range(1, data_points + 1):
        tau = 3.5 * idx / data_points
        performance_profiles[m][tau] = sum(np.log(p[m][i]) < tau for i in range(I)) / I
    return performance_profiles


def plot_performance_profiles(data: Dict[str, dict]) -> None:
    """
    Plots the performance profiles for a given dictionary of data.

    Parameters:
        data (Dict[str, dict]): A dictionary where the keys represent the names of the
            performance profiles, and the values are dictionaries containing performance profile data.

    Returns:
        None
    """

    fig, ax = plt.subplots()
    for m, pp in data.items():
        ax.plot(pp.keys(), pp.values(), label=m)

    # Remove boarder, grey background, grid lines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_facecolor("#EDEDED")
    ax.grid(color="white")
    ax.tick_params(axis="both", which="both", color="white")

    ax.set_xlabel(r"$\tau$")
    ax.set_xlim((0, max(pp.keys())))
    ax.set_ylabel(r"$P(\log(p_{im})\leq \tau)$")
    ax.set_ylim((0, 1.1))
    fig.savefig("performance_profile.pdf", format="pdf")


def aggregate_results(
    results: Dict[str, pd.DataFrame], parameter: Literal["Gamma", "Delta"], value: int
):

    # Filter results for each method by given value for given parameter.
    filtered_results = {}
    for m, data in results.items():
        filtered_results[m] = data[data[parameter] == value]

    # Compute best found solution across all solution methods
    objvals = {m: data["objval"].tolist() for m, data in filtered_results.items()}
    min_objval = [min(objval) for objval in zip(*objvals.values())]

    # Format runtimes
    for m, data in filtered_results.items():
        data_copy = data.copy()
        data_copy.loc[data["runtime"] == 0.00, "runtime"] = 0.01
        data_copy.loc[data["runtime"] > 600, "runtime"] = 600
        data_copy["solved"] = (data["status"] == 2).astype(int)
        data_copy["LBgap"] = np.where(
            data["status"] != 2, (min_objval - data["objbound"]) * 100 / min_objval, 0
        )
        data_copy["UBgap"] = np.where(
            data["status"] != 2, (data["objval"] - min_objval) * 100 / data["objval"], 0
        )
        filtered_results[m] = data_copy
        print(filtered_results[m])

    aggregate_results = {}
    for m, data in filtered_results.items():
        aggregate_results[m] = (
            data.groupby(["n", "Gamma", "Delta"])
            .agg({"runtime": "mean", "LBgap": "mean", "UBgap": "mean", "solved": "sum"})
            .round(1)
        )
        print(aggregate_results[m])

    for m, df in aggregate_results.items():
        output_dir = Path(Path.cwd() / f"../results/aggregate_by_{parameter}")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(
            output_dir,
            f"{m}_aggregate_results.csv",
        )
        # Save the DataFrame to a CSV file
        df.to_csv(filename)


if __name__ == "__main__":

    general = read_results(Path(Path.cwd() / "../results/general_results.txt"))
    general_ws = read_results(Path(Path.cwd() / "../results/general_ws_results.txt"))
    assignment = read_results(Path(Path.cwd() / "../results/assignment_results.txt"))
    assignment_ws = read_results(
        Path(Path.cwd() / "../results/assignment_ws_results.txt")
    )
    matching = read_results(Path(Path.cwd() / "../results/matching_results.txt"))
    matching_ws = read_results(Path(Path.cwd() / "../results/matching_ws_results.txt"))

    pp = get_performance_profiles(
        {
            "assignment": assignment,
            "assignment_ws": assignment_ws,
            "matching": matching,
            "matching_ws": matching_ws,
        },
        [10, 15, 20],
    )
    plot_performance_profiles(pp)

