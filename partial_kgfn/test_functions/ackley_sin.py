#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The Ackley function network test problem.
"""


from typing import Callable, List, Optional, Union

import numpy as np
import torch
from botorch.test_functions.synthetic import Ackley
from torch import Tensor

from partial_kgfn.models.dag import DAG


class AckleyFunctionNetwork(Ackley):
    n_nodes = 2
    impose_assumption = True
    node_groups = [[0], [1]]
    parent_nodes = [[], [0]]
    dag = DAG(parent_nodes=parent_nodes)
    active_input_indices = [[0, 1, 2, 3, 4, 5], []]
    node_dims = [6, 1]

    def __init__(self, node_costs: List[Union[float, int]], **kwargs) -> None:
        """Initialize the function network.

        Args:
            node_costs: cost of evaluating each of the nodes in the function network.

        Returns:
            None

        """
        self.node_costs = node_costs
        self.parent_bounds = None
        super().__init__(**kwargs)
        self.dim = 6
        self.node_dims = [6, 1]
        self.bounds = torch.tensor(
            [
                [-2.00, -2.00, -2.00, -2.00, -2.00, -2.00],
                [2.00, 2.00, 2.00, 2.00, 2.00, 2.00],
            ]
        )
        self.parent_bounds = [None, torch.Tensor([[0], [20]])]
        self.ackley = Ackley(dim=6)

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
                    if input_shape[-1] != self.node_dims[0]:
                        raise ValueError(
                            f"Mismatch dimension: Node {idx} input dimension is "
                            f"{self.node_dims[0]}, got {input_shape[-1]}"
                        )
                elif idx == [1]:
                    if input_shape[-1] != self.node_dims[1]:
                        raise ValueError(
                            f"Mismatch dimension: Node {idx} input dimension is "
                            f"{self.node_dims[1]}, got {input_shape[-1]}"
                        )

        def f0(X):
            return -1 * self.ackley(X)

        def f1(X):
            return torch.sin(5 * X / (6 * torch.pi)) * (-X)

        if idx is None:
            output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes])).to(X)
            output[..., 0] = f0(X)
            output[..., 1] = f1(output[..., 0])
        elif idx == [0]:
            output = f0(X).unsqueeze(-1)
        elif idx == [1]:
            output = f1(X)

        return output
