#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Run an experiment for a function network test problem.
"""
import argparse
import gc
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import PosteriorMean as GPPosteriorMean
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import qKnowledgeGradient as vanillaKG
from botorch.acquisition import qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective, MCAcquisitionObjective
from botorch.logging import logger
from botorch.models.model import Model
from botorch.optim import optimize_acqf as botorch_optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor, normal
from torch.distributions.multivariate_normal import MultivariateNormal

from fast_pkgfn.acquisition.full_kgfn import FullKnowledgeGradientFN
from fast_pkgfn.acquisition.partial_kgfn import PartialKnowledgeGradientFN
from fast_pkgfn.acquisition.FN_realization import FN_realization
from fast_pkgfn.acquisition.tsfn import ThompsonSamplingFN
from fast_pkgfn.models.decoupled_gp_network import (
    GaussianProcessNetwork,
    fit_gp_network,
    initialize_GP,
)
from fast_pkgfn.optim.discrete_kgfn_optim import optimize_discrete_acqf_for_function_network
from fast_pkgfn.optim.discrete_kgfn_optim import (
    optimize_discrete_acqf_for_function_network,
)
from botorch.optim.optimize import optimize_acqf
from fast_kgfn.test_functions.ack_mat import AckleyMatyasFunctionNetwork
from fast_kgfn.test_functions.ackley_sin import AckleyFunctionNetwork
from fast_kgfn.test_functions.freesolv3 import Freesolv3FunctionNetwork
from fast_kgfn.test_functions.manufacter_gp import ManufacturingGPNetwork
from fast_kgfn.test_functions.pharmaceutical import PharmaFunctionNetwork
from fast_pkgfn.utils.construct_obs_set import construct_obs_set
from fast_pkgfn.utils.EIFN_optimize_acqf import optimize_acqf_and_get_suggested_point
from fast_pkgfn.utils.gen_batch_x_fantasies import GenbatchXFantasiesFN
from fast_pkgfn.utils.posterior_mean import PosteriorMean

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


def run_one_trial(
    problem_name: str,
    problem: SyntheticTestFunction,
    algo: str,
    trial: int,
    metrics: List[str],
    n_init_evals: int,
    budget: Union[float, int],
    options: Optional[Dict] = None,
    objective: Optional[MCAcquisitionObjective] = None,
    force_restart: Optional[bool] = False,
    impose_assump: Optional[bool] = True,
    noisy: Optional[bool] = False,
) -> None:
    """Run one trial of BO loop for the given problem and algorithm.

    Args:
        problem_name: A string representing the name of the test problem.
        problem: A function network test problem.
        algo: A string representing the name of the algorithm.
        trial: The seed of the trial
        metrics: A list of metrics to record. Options are "pos_mean" and "obs_val"
        n_init_evals: Number of initial evaluations.
        budget: The budget for the BO loop.
        objective: MCAcquisitionfunctionObjective used to combine function network intemediate outputs to form a function network final value
        force_restrate: a boolean indicating to restart the experiment and ignore the exisiting result, if any
        impose_assump: a boolean indicating if the upstream-downstream restriction is imposed
        noisy: a boolean indicating if the function evaluation is noisy.

    Returns:
        None.
    """
    # options = {} or options
    # Get script directory
    results_dir = f"./results/{problem_name}_{'_'.join(str(x) for x in problem.node_costs)}/{algo}/"
    os.makedirs(results_dir, exist_ok=True)
    if objective is None:
        objective = GenericMCObjective(lambda Y: Y[..., -1])

    if os.path.exists(results_dir + f"trial_{trial}.pt") and not force_restart:
        logger.info(
            f"============================Resume Experiment=================================\n"
            f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
            f"Algorithm: {algo}\n"
            f"Trial: {trial}"
        )
        res = torch.load(results_dir + f"trial_{trial}.pt",weights_only=False)
        # reset the random seed
        torch.set_rng_state(res["random_states"]["torch"])
        np.random.set_state(res["random_states"]["numpy"])
        random.setstate(res["random_states"]["random"])
        # Get data
        train_X = res["train_X"]
        train_Y = res["train_Y"]
        best_obs_vals = res["best_obs_vals"]
        best_obs_val = best_obs_vals[-1]
        best_post_means = res["best_post_means"]
        best_design_post_mean = res["best_design_post_mean"]
        obj_at_best_designs = res["obj_at_best_designs"]
        best_post_mean = best_post_means[-1]
        runtimes = res["runtimes"]
        cumulative_costs = res["cumulative_costs"]
        node_indices = res["node_selected"]
        acqf_vals = res["acqf_val_list"]
        node_inputs = res["node_input_selected"]
        node_evals = res["node_eval_val"]
        node_candidates = res["node_candidates"]
        total_cost = cumulative_costs[-1]
        count = res["node_eval_counts"]
        network_output_at_X = res[
            "network_output_at_X"
        ]  # comment this out for old results as we did not store this
        if isinstance(problem, ManufacturingGPNetwork):
            modified_prior = True
        else:
            modified_prior = False
        if algo == "EI" or algo == "KG" or algo == "Random":
            model = initialize_GP(
                train_X=train_X,
                train_Y=train_Y,
                modified_prior=modified_prior,
                noisy=noisy,
            )
            mll = ExactMarginalLogLikelihood(
                model.likelihood,
                model,
            )
            fit_gpytorch_model(mll)
            posterior_mean_function = GPPosteriorMean(model=model)
            best_design, best_post_mean = optimize_acqf(
                acq_function=posterior_mean_function,
                bounds=problem.bounds.to(torch.double),
                q=1,
                num_restarts=10 * problem.dim,
                raw_samples=100 * problem.dim,
                options={"batch_limit": 5},
            )
            train_X_network, train_Y_network = construct_obs_set(
                X=train_X.clone(),
                Y=network_output_at_X,
                parent_nodes=problem.parent_nodes,
                active_input_indices=problem.active_input_indices,
            )
            model_network = GaussianProcessNetwork(
                train_X=train_X_network,
                train_Y=train_Y_network,
                dag=problem.dag,
                active_input_indices=problem.active_input_indices,
                node_groups=problem.node_groups,
                problem=problem,
                noisy=noisy,
            )
            fit_gp_network(model_network)
        else:
            model = GaussianProcessNetwork(
                train_X=train_X,
                train_Y=train_Y,
                dag=problem.dag,
                active_input_indices=problem.active_input_indices,
                node_groups=problem.node_groups,
                problem=problem,
                noisy=noisy,
            )
            fit_gp_network(model)
            best_design, best_post_mean = optimize_posterior_mean(
                model=model, problem=problem, objective=objective
            )

    else:
        logger.info(
            f"============================Start New Experiment=================================\n"
            f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
            f"Algorithm: {algo}\n"
            f"Trial: {trial}"
        )

        # Set manual seed
        torch.manual_seed(trial)
        np.random.seed(trial)
        random.seed(trial)
        if isinstance(problem, ManufacturingGPNetwork):
            modified_prior = True
        else:
            modified_prior = False
        # Generate initial design using SobolSampler
        X = (
            draw_sobol_samples(
                bounds=torch.Tensor(problem.bounds).to(**tkwargs),
                n=n_init_evals,
                q=1,
            )
            .squeeze(-2)
            .to(**tkwargs)
        )
        # Initialize GP network model
        network_output_at_X = problem.evaluate(X)
        if noisy:
            network_output_at_X = network_output_at_X + torch.normal(
                0, 1, size=network_output_at_X.shape
            )
        train_X, train_Y = construct_obs_set(
            X=X,
            Y=network_output_at_X,
            parent_nodes=problem.parent_nodes,
            active_input_indices=problem.active_input_indices,
        )
        if algo == "EI" or algo == "KG" or algo == "Random":
            train_X_network = [
                train_X[i].clone() for i in range(len(train_X))
            ]  # for computing posterior mean metric
            train_Y_network = [
                train_Y[i].clone() for i in range(len(train_Y))
            ]  # for computing posterior mean metric
            train_X = X
            if isinstance(problem, PharmaFunctionNetwork):
                train_Y = objective(torch.concat(train_Y, dim=-1)).unsqueeze(-1)
            else:
                train_Y = train_Y[-1]
            model = initialize_GP(
                train_X=train_X,
                train_Y=train_Y,
                modified_prior=modified_prior,
                noisy=noisy,
            )
            mll = ExactMarginalLogLikelihood(
                model.likelihood,
                model,
            )
            fit_gpytorch_model(mll)
            # The following lines are for computing posterior mean metric if we consider the one with leveraging network
            model_network = GaussianProcessNetwork(
                train_X=train_X_network,
                train_Y=train_Y_network,
                dag=problem.dag,
                active_input_indices=problem.active_input_indices,
                node_groups=problem.node_groups,
                problem=problem,
                noisy=noisy,
            )
            fit_gp_network(model_network)
        else:
            model = GaussianProcessNetwork(
                train_X=train_X,
                train_Y=train_Y,
                dag=problem.dag,
                active_input_indices=problem.active_input_indices,
                node_groups=problem.node_groups,
                problem=problem,
                noisy=noisy,
            )
            fit_gp_network(model)

        if "obs_val" in metrics:
            if algo == "EI" or algo == "KG" or algo == "Random":
                best_obs_val = train_Y.max().item()
            else:
                if isinstance(problem, PharmaFunctionNetwork):
                    if algo in ["EIFN", "TSFN", "Random", "KGFN"]:
                        best_obs_val = (
                            objective(torch.concat(train_Y, dim=-1)).max().item()
                        )
                    elif algo in ["pKGFN","fast_pKGFN"]:
                        best_obs_val = -torch.inf  # not applicable for pKGFN due to partial evaluations
                else:
                    best_obs_val = train_Y[-1].max().item()
            best_obs_vals = [best_obs_val]

        if "pos_mean" in metrics:
            if algo == "EI" or algo == "KG" or algo == "Random":
                best_design, best_post_mean = optimize_posterior_mean(
                    model=model_network,
                    problem=problem,
                    objective=objective,
                )
            else:
                best_design, best_post_mean = optimize_posterior_mean(
                    model=model,
                    problem=problem,
                    objective=objective,
                )
            obj_at_best_designs = [objective(problem.evaluate(best_design)).item()]
            best_post_means = [best_post_mean.item()]
            best_design_post_mean = [best_design]
            logger.info(
                f"Initial best posterior mean for the objective: "
                f"{best_post_mean.item():4f} at {best_design}"
                f"(Exact obj evaluation {objective(problem.evaluate(best_design)).item():4f})"
            )

        runtimes = [None]
        cumulative_costs = [None]
        node_indices = [None]
        acqf_vals = [None]
        node_inputs = [None]
        node_evals = [None]
        node_candidates = [None]
        total_cost = 0
        count = torch.zeros(len(problem.parent_nodes), dtype=int)
    print("==========================================================================")
    gen_x_fantasies_count = 0
    while total_cost < budget:
        remaining_budget = budget - total_cost
        logger.info(f"Remaining budget: {remaining_budget}")
        t0 = time.time()
        if algo == "fast_pKGFN":
            t0_fan_can = time.time()
            gen_x_fantasies_fn = GenbatchXFantasiesFN(
                model=model, num_samples=10, objective=objective
            )
            X_fantasies_batch, _ = optimize_acqf(
                acq_function=gen_x_fantasies_fn,
                q=10,
                num_restarts=10 * problem.dim,
                raw_samples=100 * problem.dim,
                bounds=problem.bounds,
                options={"batch_limit": 1},
            )

            if isinstance(problem, AckleyMatyasFunctionNetwork):
                rad = 2
            elif isinstance(problem, Freesolv3FunctionNetwork):
                rad = 0.1
            else:
                rad = 0.2

            X_fantasies_candidates_Lo = generate_X_fantasies_candidates_RaLo(
                radius=rad,
                no_random=0,
                no_local=10,
                problem=problem,
                best_design=best_design,
                include_best=True,  # always include best design to avoid negative KGFN value
            )
            X_fantasies_candidates = torch.cat(
                (X_fantasies_batch, X_fantasies_candidates_Lo), dim=0
            )
            logger.info(
                f"Generate X_fantansies_candidate takes {time.time()-t0_fan_can:.4f} seconds"
            )
            # Get suggested point
            (
                new_x,
                new_node,
                node_best_acq_vals,
                node_candidate,
            ) = get_suggested_node_and_input(
                algo=algo,
                problem=problem,
                model=model,
                best_obs_val=best_obs_vals[-1],
                best_post_mean=best_post_means[-1],
                best_design=best_design,
                objective=objective,
                remaining_budget=remaining_budget,
                impose_assump=impose_assump,
                X_fantasies_candidates=X_fantasies_candidates,
            )
        else:
            (
                new_x,
                new_node,
                node_best_acq_vals,
                node_candidate,
            ) = get_suggested_node_and_input(
                algo=algo,
                problem=problem,
                model=model,
                best_obs_val=best_obs_vals[-1],
                best_post_mean=best_post_means[-1],
                best_design=best_design,
                objective=objective,
                remaining_budget=remaining_budget,
                impose_assump=impose_assump,
            )

        t1 = time.time()
        logger.info(f"Optimizing the acquisition takes {t1 - t0:.4f} seconds")
        # The following lines separate cases for algorithms that always compute full network and ones that compute partially.
        if algo in ["Random", "EIFN", "TSFN", "EI", "KG", "KGFN"]:
            if total_cost + sum(problem.node_costs) > budget:
                break
        elif algo in ["KGFN", "pKGFN", "fast_pKGFN"]:
            eval_cost = [problem.node_costs[k] for k in new_node]
            eval_cost = sum(eval_cost)
            if total_cost + eval_cost > budget:
                break
        # Node evaluation
        new_y = problem.evaluate(X=new_x, idx=new_node)
        if noisy:
            new_y = new_y + torch.normal(0, 1, size=new_y.shape)
        # Update training data and comparison metrics
        if new_node is None:
            if algo == "Random":
                logger.info(
                    f"Evaluate the full network at input {new_x} (acqf val: N/A): {new_y}"
                )
            else:
                logger.info(
                    f"Evaluate the full network at input {new_x} (acqf val: {node_best_acq_vals:.4f}): {new_y}"
                )
            total_cost += sum(problem.node_costs)
            count = count + torch.ones(len(problem.parent_nodes), dtype=int)
            new_node = list(range(problem.n_nodes))
            new_obs_x, new_obs_y = construct_obs_set(
                X=new_x,
                Y=new_y,
                parent_nodes=problem.parent_nodes,
                active_input_indices=problem.active_input_indices,
            )
            if algo == "EI" or algo == "KG" or algo == "Random":
                train_X = torch.cat((train_X, new_x), dim=0)
                network_output_at_X = torch.cat((network_output_at_X, new_y), dim=0)
                if isinstance(problem, PharmaFunctionNetwork):
                    new_train_y = objective(torch.concat(new_obs_y, dim=-1)).unsqueeze(
                        -1
                    )
                else:
                    new_train_y = new_obs_y[-1]
                train_Y = torch.cat((train_Y, new_train_y), dim=0)
                model = initialize_GP(
                    train_X=train_X,
                    train_Y=train_Y,
                    modified_prior=modified_prior,
                    noisy=noisy,
                )
                mll = ExactMarginalLogLikelihood(
                    model.likelihood,
                    model,
                )
                fit_gpytorch_model(mll)
                train_X_network, train_Y_network = construct_obs_set(
                    X=train_X.clone(),
                    Y=network_output_at_X,
                    parent_nodes=problem.parent_nodes,
                    active_input_indices=problem.active_input_indices,
                )
                model_network = GaussianProcessNetwork(
                    train_X=train_X_network,
                    train_Y=train_Y_network,
                    dag=problem.dag,
                    active_input_indices=problem.active_input_indices,
                    node_groups=problem.node_groups,
                    problem=problem,
                )
                fit_gp_network(model_network)
            else:
                for idx in range(problem.n_nodes):
                    train_X[idx] = torch.cat((train_X[idx], new_obs_x[idx]), dim=0)
                    train_Y[idx] = torch.cat((train_Y[idx], new_obs_y[idx]), dim=0)
                    # Refit a selected node GP in Network
                    model.node_GPs[idx] = initialize_GP(
                        train_X=train_X[idx],
                        train_Y=train_Y[idx],
                        modified_prior=modified_prior,
                        noisy=noisy,
                    )
                    fit_gp_network(model, idx=idx)
        else:
            eval_cost = [problem.node_costs[k] for k in new_node]
            eval_cost = sum(eval_cost)
            idx_group = problem.node_groups.index(new_node)
            total_cost += eval_cost
            print(node_best_acq_vals[idx_group])
            logger.info(
                f"Evaluate at node {new_node} with input {new_x}"
                f"(acqf val (over cost): {node_best_acq_vals[idx_group] :.4f}): {new_y}"
            )
            idx_for_new_y = 0
            for j in new_node:
                train_X[j] = torch.cat(
                    (train_X[j], new_x), dim=0
                )  # we assume node in the same node groups take same x
                train_Y[j] = torch.cat((train_Y[j], new_y[..., [idx_for_new_y]]), dim=0)
                idx_for_new_y += 1
                count[j] += 1
                # Refit the selected node GP in the GP network model
                model.node_GPs[j] = initialize_GP(
                    train_X=train_X[j],
                    train_Y=train_Y[j],
                    modified_prior=modified_prior,
                    noisy=noisy,
                )
                fit_gp_network(model, idx=j)

        # Update comparison metrics
        if "obs_val" in metrics:
            if algo == "EI" or algo == "KG" or algo == "Random":
                best_obs_val = train_Y.max().item()
            else:
                if isinstance(problem, PharmaFunctionNetwork):
                    if algo in ["EIFN", "TSFN", "Random", "KGFN"]:
                        temp = torch.concat(train_Y, dim=-1)
                        best_obs_val = objective(temp).max().item()
                    elif algo in ["pKGFN"]:
                        best_obs_val = (
                            -torch.inf  # not applicable for pKGFN due to partial evaluations
                        )
                else:
                    best_obs_val = train_Y[-1].max().item()
            best_obs_vals.append(best_obs_val)
            # logger.info(f"Best observed objective value: {best_obs_val:.4f}")

        if "pos_mean" in metrics:
            if algo == "EI" or algo == "KG" or algo == "Random":
                best_design, best_post_mean = optimize_posterior_mean(
                    model=model_network, problem=problem, objective=objective
                )
            else:
                best_design, best_post_mean = optimize_posterior_mean(
                    model=model, problem=problem, objective=objective
                )
            obj_at_best_design = objective(problem.evaluate(best_design))
            obj_at_best_designs.append(obj_at_best_design.item())
            best_post_means.append(best_post_mean.item())
            best_design_post_mean.append(best_design)
            logger.info(
                f"Best posterior mean for the objective: "
                f"{best_post_mean.item():4f} at {best_design}"
                f"(Exact obj evaluation {obj_at_best_design.item():4f})"
            )

        logger.info(f"total cost used: {total_cost}")
        logger.info(
            "=========================================================================="
        )
        print(
            "=========================================================================="
        )
        gen_x_fantasies_count += 1
        # Store data
        runtimes.append(t1 - t0)
        cumulative_costs.append(total_cost)
        node_indices.append(new_node)
        node_inputs.append(new_x)
        node_evals.append(new_y)
        acqf_vals.append(node_best_acq_vals)
        node_candidates.append(node_candidate)

        BO_results = {
            "bo_budget": budget,
            "runtimes": runtimes,
            "cumulative_costs": cumulative_costs,
            "node_selected": node_indices,
            "node_input_selected": node_inputs,
            "node_eval_val": node_evals,
            "acqf_val_list": acqf_vals,
            "best_post_means": best_post_means,
            "best_design_post_mean": best_design_post_mean,
            "obj_at_best_designs": obj_at_best_designs,
            "best_obs_vals": best_obs_vals,
            "node_eval_counts": count,
            "node_candidates": node_candidates,
            "train_X": train_X,
            "train_Y": train_Y,
            "network_output_at_X": network_output_at_X,
            "random_states": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        }
        BO_model = {"model": model}
        torch.save(BO_results, results_dir + f"trial_{trial}.pt")
        torch.save(BO_model, results_dir + f"model_trial_{trial}.pt")  # dill


def get_suggested_node_and_input(
    algo: str,
    remaining_budget: float,
    problem: SyntheticTestFunction,
    model: Model,
    best_obs_val: Optional[float] = None,
    best_post_mean: Optional[float] = None,
    best_design: Optional[Tensor] = None,
    objective: Optional[MCAcquisitionObjective] = None,
    impose_assump: Optional[bool] = True,
    X_fantasies_candidates: Optional[Tensor] = None,
) -> Tuple[
    Tensor, Optional[int], Optional[Union[List[Tensor], Tensor]], Optional[List[Tensor]]
]:
    """Optimize an acquisition function and return suggested node(s) and input to be evaluated

    Args:
        algo (str): A string representing the name of the algorithm.
        problem (SyntheticTestFunction): A function network test problem.
        model (Model): A GaussianProcessNetwork class model
        best_obs_val (Optional[Tensor], optional): A value of current best final node output found so far. Defaults to Float.

    Returns:
        Tuple[Tensor, Optional[int], Optional[Union[List[Tensor], Tensor]], Optional[List[Tensor]]]:
        A four-element tuple containing:
            - A tensor of suggested input to be evaluated;
            - A suggested node index to be evaluated;
            - A list of acquisition function values for the optimal candidate at each node;
            - A list of tensors of the optimal candidate for each node (for KGFN only, otherwise return None).
    """
    if algo == "Random":
        new_x = (
            torch.rand([1, problem.dim]) * (problem.bounds[1] - problem.bounds[0])
            + problem.bounds[0]
        )
        return new_x, None, None, None

    elif algo == "EI":
        acquisition_function = ExpectedImprovement(model=model, best_f=best_obs_val)
        new_x, acqf_val = optimize_acqf(
            acq_function=acquisition_function,
            bounds=problem.bounds.to(torch.double),
            q=1,
            num_restarts=20,
            raw_samples=100,
            options={},
        )
        return new_x, None, acqf_val, None

    elif algo == "KG":
        acquisition_function = vanillaKG(
            model=model, num_fantasies=8, current_value=best_post_mean
        )
        new_x, acqf_val = botorch_optimize_acqf(
            acq_function=acquisition_function,
            bounds=problem.bounds.to(torch.double),
            q=1,
            num_restarts=20,
            raw_samples=100,
            options={},
        )
        return new_x, None, acqf_val, None

    elif algo == "KGFN":
        if isinstance(problem, AckleyFunctionNetwork):
            rad = 0.4
        elif isinstance(problem, Freesolv3FunctionNetwork):
            rad = 0.1
        elif isinstance(problem, AckleyMatyasFunctionNetwork):
            rad = 2
        else:
            rad = 0.2
        logger.info(f"Generating initial points for Full-KGFN with radius: {rad}.")
        X_fantasies_candidates_Lo = generate_X_fantasies_candidates_RaLo(
            radius=rad,
            no_random=0,
            no_local=10,
            problem=problem,
            best_design=best_design,
            include_best=True,
        )
        X_fantasies_candidates_TS = generate_X_fantasies_candidates_TS(
            no_points=10,
            model=model,
            problem=problem,
            objective=objective,
        )
        X_fantasies_candidates = torch.cat(
            (X_fantasies_candidates_TS, X_fantasies_candidates_Lo), dim=0
        )
        acquisition_function = FullKnowledgeGradientFN(
            d=problem.dim,
            model=model,
            problem_bounds=problem.bounds.to(torch.double),
            X_fantasies_candidates=X_fantasies_candidates,
            num_fantasies=8,
            inner_sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
            current_value=best_post_mean,
        )
        new_x, acqf_val = botorch_optimize_acqf(
            acq_function=acquisition_function,
            bounds=problem.bounds.to(torch.double),
            q=1,
            num_restarts=10 * problem.dim,
            raw_samples=100 * problem.dim,
            options={"batch_limit": 1},
        )
        return new_x, None, acqf_val, None

    elif algo == "pKGFN":
        logger.info(f"Imposing upstream-downstream restriction: {impose_assump}")
        node_candidate = []
        node_best_acq_vals = []
        if isinstance(problem, AckleyFunctionNetwork):
            rad = 0.4
        elif isinstance(problem, Freesolv3FunctionNetwork):
            rad = 0.1
        elif isinstance(problem, AckleyMatyasFunctionNetwork):
            rad = 2
        else:
            rad = 0.2
        logger.info(
            f"Generating initial points for par-KGFN (Disc) with radius: {rad}."
        )
        X_fantasies_candidates_Lo = generate_X_fantasies_candidates_RaLo(
            radius=rad,
            no_random=0,
            no_local=10,
            problem=problem,
            best_design=best_design,
            include_best=True,
        )
        X_fantasies_candidates_TS = generate_X_fantasies_candidates_TS(
            no_points=10,
            model=model,
            problem=problem,
            objective=objective,
        )
        X_fantasies_candidates = torch.cat(
            (X_fantasies_candidates_TS, X_fantasies_candidates_Lo), dim=0
        )
        for j in model.node_groups:
            eval_cost = [problem.node_costs[k] for k in j]
            eval_cost = sum(eval_cost)
            logger.info(f"Starting DKGFN optimization at node {j}")
            if remaining_budget < eval_cost:
                acqf_val = -torch.inf
                node_candidate.append([])
            else:
                t0 = time.time()
                X_evaluation_mask = torch.zeros(problem.n_nodes, dtype=int)
                X_evaluation_mask[j] = 1
                acq_function = PartialKnowledgeGradientFN(
                    model=model,
                    d=problem.dim,
                    q=1,
                    node_dim=problem.node_dims[j[0]],
                    num_fantasies=8,
                    X_fantasies_candidates=X_fantasies_candidates,
                    problem_bounds=problem.bounds,
                    objective=objective,
                    X_evaluation_mask=X_evaluation_mask,
                    inner_sampler=SobolQMCNormalSampler(torch.Size([64])),
                    current_value=best_post_mean,
                )
                candidate, acqf_val = optimize_discrete_acqf_for_function_network(
                    acq_function=acq_function,
                    bounds=problem.bounds.to(**tkwargs),
                    parent_bounds=problem.parent_bounds,
                    num_restarts=problem.dim * 10,
                    raw_samples=problem.dim * 100,
                    impose_assump=impose_assump,
                )
                del acq_function
                gc.collect()
                logger.info(
                    f"Best candidate {candidate} for node {j} with acqf_val (before divided by cost) {acqf_val:.4f}"
                )
                logger.info(
                    f"pKGFN optimization at node {j} took {time.time()-t0:.4f} seconds"
                )
                node_candidate.append(candidate)
            node_best_acq_vals.append(acqf_val / eval_cost)
        new_node_idx = np.argmax(node_best_acq_vals)
        new_x = node_candidate[new_node_idx]
        new_node = model.node_groups[new_node_idx]
        return new_x, new_node, node_best_acq_vals, node_candidate

    elif algo == "EIFN":
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acq_function = qExpectedImprovement(
            model=model,
            best_f=best_obs_val,
            sampler=qmc_sampler,
            objective=objective,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=objective,
        )
        new_x, acqf_val = optimize_acqf_and_get_suggested_point(
            acq_func=acq_function,
            bounds=problem.bounds,
            batch_size=1,
            posterior_mean=posterior_mean_function,
        )
        return new_x, None, acqf_val, None

    elif algo == "fast_pKGFN":
        if impose_assump:
            raise ValueError(f"Imposing upstream-downstream restriction: {impose_assump} - This should be FALSE!")
        # Getting candidate point from EIFN
        t0 = time.time()
        t0_EIFN = time.time()
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acq_function = qExpectedImprovement(
            model=model,
            best_f=best_post_mean,
            sampler=qmc_sampler,
            objective=objective,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=objective,
        )
        new_x, _ = optimize_acqf_and_get_suggested_point(
            acq_func=acq_function,
            bounds=problem.bounds,
            batch_size=1,
            posterior_mean=posterior_mean_function,
        )
        logger.info(
            f"Optimization EIFN to get a network candidate took {time.time()-t0_EIFN:.4f} seconds"
        )
        logger.info(f"EIFN candidate {new_x}")
        node_best_acq_vals = []
        node_candidate = []
        inner_sampler = SobolQMCNormalSampler(torch.Size([64]))
        # Generate one realization
        network_realization = FN_realization(model=model, objective=objective)
        outcomes = network_realization(new_x)
        for j in model.node_groups:
            eval_cost = [problem.node_costs[k] for k in j]
            eval_cost = sum(eval_cost)
            logger.info(f"Starting fast_pKGFN optimization at node {j}")
            t0_pKGFN_com = time.time()
            if remaining_budget < eval_cost:
                acqf_val = -torch.inf
                node_candidate.append([])
                node_best_acq_vals.append(acqf_val)
            # Define p-KGFN function
            else:
                X_evaluation_mask = torch.zeros(problem.n_nodes, dtype=int)
                X_evaluation_mask[j] = 1
                acq_function = PartialKnowledgeGradientFN(
                    model=model,
                    d=problem.dim,
                    q=1,
                    node_dim=problem.node_dims[j[0]],
                    num_fantasies=8,
                    X_fantasies_candidates=X_fantasies_candidates,
                    problem_bounds=problem.bounds,
                    objective=objective,
                    X_evaluation_mask=X_evaluation_mask,
                    inner_sampler=inner_sampler,
                    current_value=best_post_mean,
                )
                # check if it is a root node:
                if len(problem.parent_nodes[j[0]]) == 0:
                    candidate = new_x[..., problem.active_input_indices[j[0]]].to(
                        torch.double
                    )
                else:
                    if len(problem.active_input_indices[j[0]]) == 0:
                        candidate = outcomes[..., problem.parent_nodes[j[0]]]
                    else:
                        candidate = torch.cat(
                            (
                                new_x[..., problem.active_input_indices[j[0]]],
                                outcomes[..., problem.parent_nodes[j[0]]],
                            ),
                            dim=-1,
                        )
                node_candidate.append(candidate)
                logger.info(f"Node {j}'s candidate: {candidate}")
                pkgfn_value = acq_function(candidate) / eval_cost
                node_best_acq_vals.append(pkgfn_value.item())
            logger.info(
                f"fast_pKGFN optimization at node {j} took {time.time()-t0_pKGFN_com} secs"
            )
        new_node_idx = np.argmax(node_best_acq_vals)
        new_x_final = node_candidate[new_node_idx]
        new_node = model.node_groups[new_node_idx]
        return new_x_final, new_node, node_best_acq_vals, node_candidate

    elif algo == "TSFN":
        acq_function = ThompsonSamplingFN(model=model, objective=objective)
        new_x, acqf_val = optimize_acqf(
            acq_function=acq_function,
            bounds=problem.bounds,
            q=1,
            num_restarts=10 * problem.dim,
            raw_samples=100 * problem.dim,
            options={"batch_limit": 1},
        )
        return new_x, None, acqf_val, None


def optimize_posterior_mean(
    model: Model,
    problem: SyntheticTestFunction,
    objective: Optional[GenericMCObjective] = None,
):
    """Optimize the posterior mean of the model.

    Args:
        model: A Gaussian process network model.
        problem: A synthetic test function problem instance.

    Returns:
        A two-element tuple containing
        - The design that achieves the best posterior mean for the objective.
        - The best posterior mean for the objective
    """
    qSR = qSimpleRegret(
        model=model,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512])),
        objective=objective,
    )
    best_design, best_post_mean = optimize_acqf(
        acq_function=qSR,
        bounds=problem.bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        return_best_only=True,
    )
    return best_design, best_post_mean


def generate_X_fantasies_candidates_RaLo(
    radius: float,
    no_random: int,
    no_local: int,
    problem: SyntheticTestFunction,
    best_design: Tensor,
    include_best: bool = True,
):
    if no_random == 0 and no_local == 0:
        X_fantasies_candidates = best_design
        return X_fantasies_candidates
    if no_random != 0:
        X_fantasies_candidates_random = draw_sobol_samples(
            bounds=problem.bounds.to(**tkwargs),
            n=no_random,
            q=1,
        ).squeeze(-2)
    if no_local != 0:
        mdist = MultivariateNormal(
            loc=torch.zeros(problem.dim), covariance_matrix=torch.eye(problem.dim)
        )
        random_directions = mdist.sample([no_local])
        random_directions = torch.transpose(
            torch.transpose(random_directions, 0, 1) / random_directions.norm(dim=1),
            0,
            1,
        )
        random_radii = torch.rand([no_local]) ** (1 / problem.dim)
        random_radii = random_radii.unsqueeze(dim=1)
        X_fantasies_candidates_local = (
            radius * (random_directions * random_radii) + best_design
        )

        if no_random != 0:
            X_fantasies_candidates = torch.cat(
                (X_fantasies_candidates_random, X_fantasies_candidates_local),
                dim=0,
            )
        else:
            X_fantasies_candidates = X_fantasies_candidates_local
    else:
        X_fantasies_candidates = X_fantasies_candidates_random
    if include_best:
        X_fantasies_candidates = torch.cat((X_fantasies_candidates, best_design), dim=0)
    for i in range(problem.dim):
        X_fantasies_candidates[..., i][
            X_fantasies_candidates[..., i] < problem.bounds[0, i]
        ] = problem.bounds[0, i].item()
        X_fantasies_candidates[..., i][
            X_fantasies_candidates[..., i] > problem.bounds[1, i]
        ] = problem.bounds[1, i].item()
    return X_fantasies_candidates


def generate_X_fantasies_candidates_TS(
    no_points: int,
    model: Model,
    problem: SyntheticTestFunction,
    objective: Optional[MCAcquisitionObjective] = None,
):
    X_fantasies_candidates = torch.Tensor([])
    for i in range(no_points):
        acq_function = ThompsonSamplingFN(model=model, objective=objective)
        new_x, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=problem.bounds,
            q=1,
            num_restarts=10 * problem.dim,
            raw_samples=100 * problem.dim,
            options={"batch_limit": 1},
        )
        X_fantasies_candidates = torch.cat((X_fantasies_candidates, new_x), dim=0)
    return X_fantasies_candidates


def generate_X_fantasies_candidates_Greedy(
    no_total_TS: int,
    no_points: int,
    model: Model,
    problem: SyntheticTestFunction,
    objective: Optional[MCAcquisitionObjective] = None,
):
    X_fantasies_candidates_TS = torch.Tensor([])
    values = []
    sample_f = []
    for i in range(no_total_TS):
        acq_function = ThompsonSamplingFN(model=model, objective=objective)
        sample_f.append(acq_function)  # @TODO check if this works?
        new_x, val = optimize_acqf(
            acq_function=acq_function,
            bounds=problem.bounds,
            q=1,
            num_restarts=10 * problem.dim,
            raw_samples=100 * problem.dim,
            options={"batch_limit": 1},
        )
        X_fantasies_candidates_TS = torch.cat((X_fantasies_candidates_TS, new_x), dim=0)
        values.append(val)
    X_fantasies_candidates = torch.Tensor([])
    constant = np.mean(values)
    while X_fantasies_candidates.shape[0] < no_points:
        if X_fantasies_candidates.shape[0] == 0:
            val_candidate = [
                np.mean([func(X_fantasies_candidates_TS[i]) for func in sample_f])
                for i in range(no_total_TS)
            ]
            idx = np.argmax(val_candidate)
            X_fantasies_candidates = torch.cat(
                (X_fantasies_candidates, X_fantasies_candidates_TS[[idx], ...]), dim=0
            )
            X_fantasies_candidates_TS = torch.cat(
                (
                    X_fantasies_candidates_TS[:idx, ...],
                    X_fantasies_candidates_TS[idx + 1 :, ...],
                ),
                dim=0,
            )
        else:
            val_candidate = [
                np.mean(
                    [
                        torch.max(
                            func(
                                torch.cat(
                                    (
                                        X_fantasies_candidates,
                                        X_fantasies_candidates_TS[i],
                                    ),
                                    dim=0,
                                )
                            )
                        )
                        for func in sample_f
                    ]
                )
                for i in range(X_fantasies_candidates_TS.shape[0])
            ]
            val_candidate = [constant - value for value in val_candidate]
            idx = np.argmax(val_candidate)
            X_fantasies_candidates = torch.cat(
                (X_fantasies_candidates, X_fantasies_candidates_TS[[idx], ...]), dim=0
            )
            X_fantasies_candidates_TS = torch.cat(
                (
                    X_fantasies_candidates_TS[:idx, ...],
                    X_fantasies_candidates_TS[idx + 1 :, ...],
                ),
                dim=0,
            )
    return X_fantasies_candidates


def parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one replication of a BO experiment."
    )
    parser.add_argument("--trial", "-t", type=int, default=0)
    parser.add_argument("--algo", "-a", type=str, default="KGFN")
    parser.add_argument("--costs", "-c", type=str, required=True)
    parser.add_argument("--budget", "-b", type=int, default=200)
    return parser.parse_args()
