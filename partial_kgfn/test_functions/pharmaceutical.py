#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The pharmaceutical function network test problem.
"""


from typing import List, Optional, Union

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor

from partial_kgfn.models.dag import DAG


class PharmaFunctionNetwork(SyntheticTestFunction):
    """Function network test problem based on the DropWave function."""

    impose_assumption = True
    parent_nodes = [[] for _ in range(2)]
    dag = DAG(parent_nodes=parent_nodes)
    active_input_indices = [[0, 1, 2, 3] for _ in range(2)]
    n_nodes = 2
    node_dims = [4 for _ in range(2)]
    node_groups = [[i] for i in range(2)]
    dim = 4
    _bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]

    def __init__(
        self, node_costs: List[Union[float, int]], negate: bool = True, **kwargs
    ) -> None:
        """Initialize the function network.

        Args:
            node_costs: cost of evaluating each of the nodes in the function network.

        Returns:
            None
        """
        self._optimizers = [
            tuple((-1.0, 0.08995383, 0.15457972, -0.51531699))
        ]  # from scipy min
        self.node_costs = node_costs
        self.parent_bounds = None
        super().__init__(**kwargs)

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

        def TS(X):
            term1 = 0.62 * (
                1
                + torch.exp(
                    -(
                        3.05
                        + 0.03 * X[..., 0]
                        - 0.16 * X[..., 1]
                        + 4.03 * X[..., 2]
                        - 0.54 * X[..., 3]
                    )
                )
            ) ** (-1)
            term2 = 0.65 * (
                1
                + torch.exp(
                    -(
                        1.78
                        + 0.60 * X[..., 0]
                        - 3.19 * X[..., 1]
                        + 0.10 * X[..., 2]
                        + 0.52 * X[..., 3]
                    )
                )
            ) ** (-1)
            term3 = -0.72 * (
                1
                + torch.exp(
                    -(
                        0.01
                        + 2.04 * X[..., 0]
                        - 3.73 * X[..., 1]
                        + 0.10 * X[..., 2]
                        - 1.05 * X[..., 3]
                    )
                )
            ) ** (-1)
            term4 = -0.45 * (
                1
                + torch.exp(
                    -(
                        1.82
                        + 4.78 * X[..., 0]
                        + 0.48 * X[..., 1]
                        - 4.68 * X[..., 2]
                        - 1.65 * X[..., 3]
                    )
                )
            ) ** (-1)
            term5 = -0.32 * (
                1
                + torch.exp(
                    -(
                        2.69
                        + 5.99 * X[..., 0]
                        + 3.87 * X[..., 1]
                        + 3.10 * X[..., 2]
                        - 2.17 * X[..., 3]
                    )
                )
            ) ** (-1)
            return 1.07 + term1 + term2 + term3 + term4 + term5

        def DT(X):
            term1 = 9.20 * (
                1
                + torch.exp(
                    -(
                        0.32
                        + 5.06 * X[..., 0]
                        - 4.07 * X[..., 1]
                        - 0.36 * X[..., 2]
                        - 0.34 * X[..., 3]
                    )
                )
            ) ** (-1)
            term2 = 9.88 * (
                1
                + torch.exp(
                    -(
                        -4.83
                        + 7.43 * X[..., 0]
                        + 3.46 * X[..., 1]
                        + 9.19 * X[..., 2]
                        + 16.58 * X[..., 3]
                    )
                )
            ) ** (-1)
            term3 = 10.84 * (
                1
                + torch.exp(
                    -7.90
                    - 7.91 * X[..., 0]
                    - 4.48 * X[..., 1]
                    - 4.08 * X[..., 2]
                    - 8.28 * X[..., 3]
                )
            ) ** (-1)
            term4 = 15.18 * (
                1
                + torch.exp(
                    -(
                        9.41
                        - 7.99 * X[..., 0]
                        + 0.65 * X[..., 1]
                        + 3.14 * X[..., 2]
                        + 0.31 * X[..., 3]
                    )
                )
            ) ** (-1)
            return -3.95 + term1 + term2 + term3 + term4

        if idx is None:
            output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes])).to(X)
            output[..., 0] = DT(X)
            output[..., 1] = TS(X)
        elif idx == [0]:
            output = DT(X).unsqueeze(-1)
        elif idx == [1]:
            output = TS(X).unsqueeze(-1)
        return output
