#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The FreeSolv function network test problem.
"""

import os
from typing import List, Optional, Union

import pandas as pd
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import PosteriorMean
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from botorch.test_functions.synthetic import SyntheticTestFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from partial_kgfn.models.dag import DAG
from partial_kgfn.models.decoupled_gp_network import (
    GaussianProcessNetwork,
    fit_gp_network,
    initialize_GP,
)


class Freesolv3FunctionNetwork(SyntheticTestFunction):
    """Function network test problem based on the FreeSolv function."""

    impose_assumption = True
    subtree = [[[0], [1]], [[1]]]
    node_fixed_attribute = {"[[0], [1]]": [], "[[1]]": [[0]]}  # List[List[int]]
    num_fantasies = {"[[0], [1]]": [8, 1], "[[1]]": [8]}  # List[int]
    subtree_node_costs = {"[[0], [1]]": [0, 1], "[[1]]": [1]}  # List[int]
    parent_nodes = [[], [0]]
    dag = DAG(parent_nodes=parent_nodes)
    active_input_indices = [[i for i in range(3)], []]
    n_nodes = 2
    dim = 3
    node_dims = [3, 1]
    node_groups = [[0], [1]]

    # Specify the main and alternative directories
    loading_directory = "./partial_kgfn/test_functions/"

    data = pd.read_csv(f"{loading_directory}freesolv_NN_rep3dim.csv")
    data_tensor = torch.tensor(data.values)
    train_X = [data_tensor[..., :3], data_tensor[..., [3]]]
    train_Y = [data_tensor[..., [3]], data_tensor[..., [4]]]
    model0 = SingleTaskGP(
        train_X=data_tensor[..., :3],
        train_Y=data_tensor[..., [3]],
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=dim),
    )
    node_mll_0 = ExactMarginalLogLikelihood(
        model0.likelihood,
        model0,
    )
    fit_gpytorch_model(node_mll_0)
    model1 = SingleTaskGP(
        train_X=data_tensor[..., [3]],
        train_Y=data_tensor[..., [4]],
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=1),
    )
    node_mll_1 = ExactMarginalLogLikelihood(
        model1.likelihood,
        model1,
    )
    fit_gpytorch_model(node_mll_1)
    f_node_0 = PosteriorMean(model0)
    f_node_1 = PosteriorMean(model1)
    bounds = torch.Tensor([[0 for _ in range(3)], [1 for _ in range(3)]]).to(
        torch.double
    )

    def __init__(self, node_costs: List[Union[float, int]], **kwargs) -> None:
        """Initialize the function network.

        Args:
            node_costs: cost of evaluating each of the nodes in the function network.

        Returns:
            None
        """
        self.parent_bounds = None
        self.node_costs = node_costs
        self.parent_bounds = [
            None,
            torch.Tensor([[-5], [30]]),
        ]

    def evaluate_true(self, X: Tensor) -> None:
        return None

    def evaluate(
        self,
        X: Tensor,
        idx: Optional[int] = None,
    ) -> Tensor:
        """Evaluate the function network.

        Args:
            X: input tensor to evaluate the function network at.
            idx: index of the node to evaluate. If None, evaluate all nodes.

        Returns:
            A tensor of shape `X.shape[:-1] + torch.Size([self.n_nodes])` if `idx` is None,
            or a tensor of shape `X.shape[:-1] + torch.Size([1])` otherwise.
        """
        if X.ndim == 2 and X.shape[0] != 1:
            X = X.unsqueeze(-2)
        input_shape = X.shape

        if idx is None:
            if input_shape[-1] != self.dim:
                raise ValueError(
                    f"Mismatch dimension: Full eval dimension is {self.dim}, "
                    f"got {input_shape[-1]}"
                )
        else:
            if idx not in self.node_groups:
                raise ValueError(f"Invalid node index {idx}")
            else:
                if idx == [0]:
                    if input_shape[-1] != self.dim:
                        raise ValueError(
                            f"Mismatch dimension: Node {idx} input dimension is "
                            f"{self.dim}, got {input_shape[-1]}"
                        )
                elif idx == [1]:
                    if input_shape[-1] != self.node_dims[1]:
                        raise ValueError(
                            f"Mismatch dimension: Node {idx} input dimension is "
                            f"{self.node_dims[1]}, got {input_shape[-1]}"
                        )

        def f0(X):
            return self.f_node_0(X).unsqueeze(-1).detach()

        def f1(X):
            return self.f_node_1(X).unsqueeze(-1).detach()

        if idx is None:
            output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes])).to(X)
            output[..., 0] = f0(X)
            output[..., 1] = f1(output[..., [0]])
        elif idx == [0]:
            output = f0(X).unsqueeze(-1)
        elif idx == [1]:
            output = f1(X).unsqueeze(-1)
        if output.ndim == 3:
            output = output.squeeze(1)
        return output
