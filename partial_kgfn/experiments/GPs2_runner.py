import warnings

import torch
from botorch.acquisition.objective import (
    GenericMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.settings import debug

from partial_kgfn.models.dag import DAG
from partial_kgfn.run_one_trial import parse, run_one_trial
from partial_kgfn.test_functions.GPs2 import GPs2

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
    impose_assump: bool = False,
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
        "1_1_1_47": [1, 1, 1, 47],
    }
    if costs not in cost_options:
        raise ValueError(f"Invalid cost option: {costs}")
    problem = GPs2(node_costs=cost_options[costs])
    problem_name = "GPs2"
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
        noisy=noisy,
        impose_assump=impose_assump,
    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))
