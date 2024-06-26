import warnings

import torch
from botorch.acquisition.objective import (
    GenericMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.settings import debug
from memory_profiler import profile

from decoupled_kg.models.dag import DAG
from decoupled_kg.run_one_trial import parse, run_one_trial
from decoupled_kg.test_functions.GPs1 import GPs1

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
debug._set_state(True)
import logging

logger = logging.getLogger("botorch")
logger.setLevel(logging.INFO)
logger.handlers.pop()


@profile
def main(
    trial: int,
    algo: str,
    costs: str,
    budget: int,
) -> None:
    """Run one replication for the dropwave function network test problem

    Args:
        trial: Seed of the trial.
        algo: Algorithm to use. Supported algorithms: "KGFN", "EIFN", "Random".
        costs: A str indicating the costs of evaluating the nodes in the network.
        budget: The total budget of the BO loop.

    Returns:
        None.
    """
    # construct the problem
    cost_options = {
        "1_49": [1, 49],
        "1_1": [1, 1],
        "1_9": [1, 9],
    }
    if costs not in cost_options:
        raise ValueError(f"Invalid cost option: {costs}")
    problem = GPs1(node_costs=cost_options[costs])
    problem_name = "GPs1"
    network_objective = GenericMCObjective(lambda Y: Y[..., -1])
    # set comparison metrics
    metrics = ["obs_val", "pos_mean"]  # obs_val  pos_mean
    run_one_trial(
        problem_name=problem_name,
        problem=problem,
        algo=algo,
        trial=trial,
        metrics=metrics,
        n_init_evals=2 * problem.dim + 1,
        budget=budget,
        objective=network_objective,
    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))
