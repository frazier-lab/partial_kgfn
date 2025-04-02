#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Methods for optimizing decoupled acquisition functions.
"""
import itertools
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.logging import logger
from torch import Tensor

from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from partial_pkgfn.models.decoupled_gp_network import GaussianProcessNetwork
from botorch.optim.optimize import optimize_acqf


def optimize_discrete_acqf_for_function_network(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    parent_bounds: Optional[List[Tensor]] = None,
    return_best_only: bool = True,
    impose_assump: bool = True,
    **kwargs: Any,
):
    r"""Generate a set of candidates for a GP function network using
    a decoupled acquisition function.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` Tensor of lower and upper bounds for the input
            of the GP network.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization.
        parent_bounds: Optional[List[Tensor]]: List of tensors indicating bounds of parent nodes' outputs 
            for each node. This will be used by p-KGFN when upstream-downstream assumption is not imposed.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        return full tree: Whether to return the full tree consisting of `X_actual`
            and `X_fantasies` for a OneShotAcquisitionFunction
        impose_assump: A boolean variable indicating if the problem being considered has the 
            upstream-downstream assumption imposed.

    Returns:
        TODO

    """
    if impose_assump:
        # We put [0] in node_idx[0] because we assume that the nodes in the node_group take the same input
        node_idx = (acq_function.X_evaluation_mask == 1).nonzero()
        parent_nodes = acq_function.model.dag.get_parent_nodes(node_idx[0])
        active_indices = acq_function.model.active_input_indices[node_idx[0]]
        bounds = bounds[:, active_indices]
        # Case 1: root node
        if len(parent_nodes) == 0:
            candidates, acq_vals = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                return_full_tree=False,
                return_best_only=return_best_only,
                options={
                    "batch_limit": 1,
                    "init_batch_limit": raw_samples,
                },  # **kwargs,
            )
            return candidates, acq_vals.item()
        # Case 2: non-root nodes
        num_active_inputs = len(acq_function.model.active_input_indices[node_idx[0]])
        node_inputs = retrieve_available_evaluations(
            acq_function.model, node_idx[0]
        ).to(
            torch.double
        )  # We pass only the representative (singleton) of nodes in the node group.
        if node_inputs is None:
            return None, -torch.inf
        column_fixed = node_inputs.shape[-1]
        batched_acq_vals = []
        batched_candidates = []
        for i in range(node_inputs.shape[0]):
            logger.debug(
                f"fixed dims: {list(range(acq_function.node_dim - column_fixed, acq_function.node_dim))}; "
                f"fixed node input: {node_inputs[i]}"
            )
            if num_active_inputs == 0:
                if (
                    torch.isclose(node_inputs[i], acq_function.model.train_X[node_idx])
                    .all(-1)
                    .any(-1)
                ):
                    # if node_inputs[i] in acq_function.model.train_X[node_idx]:
                    logger.info(
                        f"Node input {node_inputs[i]} has been evaluated. "
                        f"Skip KGFN optimization problem {i}!"
                    )
                    # skip optimizing an acqf for fixing an input that has been evaluated
                    batched_candidates.append(node_inputs[i])
                    batched_acq_vals.append(None)
                    continue
                else:
                    # Optimize only X_fantasies with fixing X_actual. Use evaluate function instead.
                    acq_vals = acq_function.forward(node_inputs[i].unsqueeze(0))
            else:
                acqf_fixed_node_inputs = FixedFeatureAcquisitionFunction(
                    acq_function=acq_function,
                    d=acq_function.q * acq_function.node_dim,
                    columns=list(
                        range(
                            acq_function.node_dim - column_fixed, acq_function.node_dim
                        )
                    ),
                    values=node_inputs[i],
                )
                logger.info(f"Acquisition optimization problem {i} for node {node_idx}")
                t0 = time.time()
                candidates, acq_vals = optimize_acqf(
                    acq_function=acqf_fixed_node_inputs,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    return_best_only=False,
                    options={"batch_limit": 1, "init_batch_limit": raw_samples},
                    # **kwargs,
                )
                logger.info(
                    f"Node 1 problem {i} optimization takes {time.time() - t0:.4f} seconds"
                )
            if num_active_inputs == 0:
                candidates = (
                    node_inputs[i].unsqueeze(0).unsqueeze(0).repeat(num_restarts, 1, 1)
                )
            else:
                candidates = torch.cat(
                    (
                        candidates,
                        torch.ones(candidates.shape[:-1] + torch.Size([1]))
                        * node_inputs[i],
                    ),
                    dim=-1,
                )
            batched_candidates.append(candidates)
            batched_acq_vals.append(acq_vals)
            logger.debug(
                f"max acqf val: {acq_vals.max():.4f} for fixed node input: {node_inputs[i]}"
            )
    else:
        node_idx = (acq_function.X_evaluation_mask == 1).nonzero()
        parent_nodes = acq_function.model.dag.get_parent_nodes(node_idx[0])
        active_indices = acq_function.model.active_input_indices[node_idx[0]]
        if parent_bounds[node_idx[0]] is not None:
            all_bounds = torch.cat(
                (
                    bounds[:, active_indices],
                    parent_bounds[node_idx[0]].to(torch.double),
                ),
                dim=-1,
            ).to(torch.double)
        else:
            all_bounds = bounds[:, active_indices]
        candidates, acq_vals = optimize_acqf(
            acq_function=acq_function,
            bounds=all_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            return_full_tree=False,
            return_best_only=return_best_only,
            options={
                "batch_limit": 1,
                "init_batch_limit": raw_samples,
            },  # **kwargs,
        )
        return candidates, acq_vals.item()

    if return_best_only:
        # exclude None's
        valid_indices = [
            i for i, acq_vals in enumerate(batched_acq_vals) if acq_vals is not None
        ]
        if not valid_indices:
            return None, -torch.inf

        batched_acq_vals = [batched_acq_vals[i] for i in valid_indices]
        batched_candidates = [batched_candidates[i] for i in valid_indices]
        # max across restarts
        arg_max_idx = [torch.argmax(acq_vals) for acq_vals in batched_acq_vals]
        batched_acq_vals = [
            acq_vals[idx] for acq_vals, idx in zip(batched_acq_vals, arg_max_idx)
        ]
        batched_candidates = [
            candidates[idx] for candidates, idx in zip(batched_candidates, arg_max_idx)
        ]

        # max across node inputs
        arg_max_idx = np.argmax(
            [acq_val.detach().numpy() for acq_val in batched_acq_vals]
        )
        batched_candidates = batched_candidates[arg_max_idx]
        batched_acq_vals = batched_acq_vals[arg_max_idx].item()
    logger.info(
        f"candidate {batched_candidates} with acqf_vals (before divided by cost) {batched_acq_vals:.4f}"
    )

    return batched_candidates, batched_acq_vals


def retrieve_available_evaluations(
    model: GaussianProcessNetwork,
    node_idx: int,
) -> Optional[Tensor]:
    r"""Retrieve information about available evaluations for a node in the function network.

    Args:
        model: A GaussianProcessNetwork model.
        node_idx: The index of the node for which to retrieve available evaluations.

    Returns:
        A `n x d`-dim tensor of available evaluations for node `node_idx`, where `n` is the
        product of the number of available evaluations for each of the parent nodes of
        node `node_idx`, and `d` is the number of parent nodes of node `node_idx`.
    """
    parent_nodes = model.dag.get_parent_nodes(node_idx)  # node_idx is a singleton
    Ys = []
    for group in model.node_groups:
        if set(group).issubset(set(parent_nodes)):
            Y_group = [model.train_Y[i] for i in group]
            Y_group = torch.cat(Y_group, dim=-1)
            Ys.append(Y_group.tolist())
    if len(Ys) == 0 or any(Y is None for Y in Ys):
        return None

    if len(Ys) == 1:
        return Y_group
        # return torch.Tensor(Ys[0])

    res = []
    for element in itertools.product(*Ys):
        res.append(sum(element, []))
    output = torch.Tensor(res)
    return output
