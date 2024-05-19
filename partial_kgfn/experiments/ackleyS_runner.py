#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug

from partial_kgfn.run_one_trial import parse, run_one_trial
from partial_kgfn.test_functions.ackley_sin import AckleyFunctionNetwork

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
debug._set_state(True)
import logging

logger = logging.getLogger("botorch")
logger.setLevel(logging.INFO)
logger.handlers.pop()


def main(
    trial: int,
    algo: str,
    costs: str,
    budget: int,
    noisy: bool = False,
) -> None:
    """Run one replication for the Ackley test problem.

    Args:
        trial: Seed of the trial.
        algo: Algorithm to use. Supported algorithms: "EI", "KG", "Random", "EIFN", "KGFN", "TSFN", "pKGFN";
        costs: A str indicating the costs of evaluating the nodes in the network.
        budget: The total budget of the BO loop.
        noisy: A boolean variable indicating if the evaluations are noisy.

    Returns:
        None.
    """
    cost_options = {
        "1_1": [1, 1],
        "1_9": [1, 9],
        "1_49": [1, 49],
    }
    noisy = True
    if costs not in cost_options:
        raise ValueError(f"Invalid cost option: {costs}")
    problem = AckleyFunctionNetwork(node_costs=cost_options[costs])
    if noisy:
        problem_name = "AckSN"
    else:
        problem_name = "AckS"
    network_objective = GenericMCObjective(lambda Y: Y[..., -1])
    metrics = ["obs_val", "pos_mean"] 
    run_one_trial(
        problem_name=problem_name,
        problem=problem,
        algo=algo,
        trial=trial,
        metrics=metrics,
        n_init_evals=2 * problem.dim + 1,
        budget=budget,
        objective=network_objective,
        noisy=noisy,
    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))
