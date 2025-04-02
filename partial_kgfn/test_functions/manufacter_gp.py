#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The Manufacturing function network test problem.
"""

from typing import Callable, List, Optional, Union

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.gp_sampling import get_gp_samples
from torch import Tensor

from partial_kgfn.models.dag import DAG


class ManufacturingGPNetwork(SyntheticTestFunction):
    n_nodes = 4
    node_groups = [[0], [1], [2], [3]]
    parent_nodes = [[], [0], [], [1, 2]]
    dag = DAG(parent_nodes=parent_nodes)
    _optimal_value = -0.9844497165719973  # max
    X0 = torch.empty(0, 2).to(torch.double)
    Y0 = torch.empty(0, 1).to(torch.double)
    model0 = SingleTaskGP(train_X=X0, train_Y=Y0)
    torch.manual_seed(88)
    function0 = get_gp_samples(model=model0, num_outputs=1, n_samples=1)

    X1 = torch.empty(0, 1).to(torch.double)
    Y1 = torch.empty(0, 1).to(torch.double)
    model1 = SingleTaskGP(train_X=X1, train_Y=Y1)
    model1.covar_module.base_kernel.lengthscale = torch.Tensor([[1]]).to(torch.double)
    torch.manual_seed(112)
    function1 = get_gp_samples(model=model1, num_outputs=1, n_samples=1)

    X2 = torch.empty(0, 2).to(torch.double)
    Y2 = torch.empty(0, 1).to(torch.double)
    model2 = SingleTaskGP(train_X=X2, train_Y=Y2)
    model2.covar_module.base_kernel.lengthscale = torch.Tensor([[1, 1]]).to(
        torch.double
    )
    torch.manual_seed(11)
    function2 = get_gp_samples(model=model2, num_outputs=1, n_samples=1)

    X3 = torch.empty(0, 2).to(torch.double)
    Y3 = torch.empty(0, 1).to(torch.double)
    model3 = SingleTaskGP(train_X=X3, train_Y=Y3)
    model3.covar_module.base_kernel.lengthscale = torch.Tensor([[3, 3]]).to(
        torch.double
    )
    model3.covar_module.outputscale = torch.tensor(10).to(torch.double)
    torch.manual_seed(185)
    function3 = get_gp_samples(model=model3, num_outputs=1, n_samples=1)

    def __init__(self, node_costs: List[Union[float, int]], **kwargs) -> None:
        """Initialize the function network.

        Args:
            node_costs: cost of evaluating each of the nodes in the function network.

        Returns:
            None

        """
        self.dim = 4
        self.node_dims = [2, 1, 2, 2]
        self._bounds = [(-1.0, 1.0) for _ in range(4)]
        self.parent_bounds = [
            None,
            torch.Tensor([[-2], [2]]),
            None,
            torch.Tensor([[-1, -1], [1, 1]]),
        ]
        # self._optimizers = [
        #     tuple((-0.48296594619750977, -0.9118236303329468, 1.0, 0.3587174415588379))
        # ]
        self.optimizers = tuple((0.4509, 0.4549, -1.0000, -0.6754))
        self.active_input_indices = [[0, 1], [], [2, 3], []]
        self.node_costs = node_costs
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
                elif idx == [2]:
                    if input_shape[-1] != self.node_dims[2]:
                        raise ValueError(
                            f"Mismatch dimension: Node {idx} input dimension is "
                            f"{self.node_dims[2]}, got {input_shape[-1]}"
                        )
                elif idx == [3]:
                    if input_shape[-1] != self.node_dims[3]:
                        raise ValueError(
                            f"Mismatch dimension: Node {idx} input dimension is "
                            f"{self.node_dims[3]}, got {input_shape[-1]}"
                        )

        def f_0(X):
            return self.function0(X)

        def f_1(X):
            return self.function1(X)

        def f_2(X):
            return self.function2(X)

        def f_3(X):
            return self.function3(X)

        if idx is None:
            output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes])).to(torch.double)
            output[..., [0]] = f_0(X[..., [0, 1]])
            output[..., [1]] = f_1(output[..., [0]])
            output[..., [2]] = f_2(X[..., [2, 3]])
            output[..., [3]] = f_3(output[..., [1, 2]])
        elif idx == [0]:
            output = f_0(X)
        elif idx == [1]:
            output = f_1(X)
        elif idx == [2]:
            output = f_2(X)
        elif idx == [3]:
            output = f_3(X)

        return output
