#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The Ackley-Matyas function network test problem.
"""


from typing import List, Optional, Union

import torch
from botorch.test_functions.synthetic import Ackley, SyntheticTestFunction
from torch import Tensor

from partial_pkgfn.models.dag import DAG


class AckleyMatyasFunctionNetwork(SyntheticTestFunction):
    n_nodes = 2
    node_groups = [[0], [1]]
    parent_nodes = [[], [0]]
    dag = DAG(parent_nodes=parent_nodes)

    def __init__(
        self, node_costs: List[Union[float, int]], dim: int = 6, **kwargs
    ) -> None:
        """Initialize the function network.

        Args:
            node_costs: cost of evaluating each of the nodes in the function network.

        Returns:
            None

        """
        self.dim = 7
        self.node_dims = [6, 2]
        self._bounds = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-10, 10)]
        self.parent_bounds = [None, torch.Tensor([[0], [20]])]
        self.active_input_indices = [
            [0, 1, 2, 3, 4, 5],
            [6],
        ]
        self.node_costs = node_costs
        super().__init__(**kwargs)
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

        def f_0(X):
            return self.ackley(X)

        def f_1(X):
            return (
                -0.26 * (X[..., 0] ** 2 + X[..., 1] ** 2) + 0.48 * X[..., 0] * X[..., 1]
            )

        if idx is None:
            output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes])).to(X)
            output[..., 0] = f_0(X[..., :6])
            output[..., 1] = f_1(torch.cat((X[..., [6]], output[..., [0]]), dim=-1))
        elif idx == [0]:
            output = f_0(X).unsqueeze(-1)
        elif idx == [1]:
            output = f_1(X).unsqueeze(-1)
        return output
