from typing import Callable, List, Optional, Union

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.gp_sampling import get_gp_samples
from torch import Tensor

from partial_kgfn.models.dag import DAG


class GPs1(SyntheticTestFunction):
    n_nodes = 2
    impose_assumption = True
    node_groups = [[0], [1]]
    parent_nodes = [[], [0]]
    dag = DAG(parent_nodes=parent_nodes)

    _optimal_value = -0.9844497165719973  # max
    X0 = torch.empty(0, 1).to(torch.double)
    Y0 = torch.empty(0, 1).to(torch.double)
    model0 = SingleTaskGP(train_X=X0, train_Y=Y0)
    model0.covar_module.base_kernel.lengthscale = torch.Tensor([[0.5]]).to(torch.double)
    torch.manual_seed(22)
    function0 = get_gp_samples(model=model0, num_outputs=1, n_samples=1)

    X1 = torch.empty(0, 1).to(torch.double)
    Y1 = torch.empty(0, 1).to(torch.double)
    model1 = SingleTaskGP(train_X=X1, train_Y=Y1)
    model1.covar_module.base_kernel.lengthscale = torch.Tensor([[0.25]]).to(
        torch.double
    )
    torch.manual_seed(33)
    function1 = get_gp_samples(model=model1, num_outputs=1, n_samples=1)

    def __init__(self, node_costs: List[Union[float, int]], **kwargs) -> None:
        """Initialize the function network.

        Args:
            node_costs: cost of evaluating each of the nodes in the function network.

        Returns:
            None

        """
        self.dim = 1
        self.node_dims = [1, 1]
        self._bounds = [(-1.0, 1.0) for _ in range(1)]
        self.parent_bounds = [
            None,
            torch.Tensor([[-1], [2]]),
        ]
        self.optimizers = 0.459
        self.active_input_indices = [[0], []]
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
            return self.function0(X)

        def f_1(X):
            return self.function1(X)

        if idx is None:
            output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes])).to(torch.double)
            output[..., [0]] = f_0(X[..., [0]])
            output[..., [1]] = f_1(output[..., [0]])

        elif idx == [0]:
            output = f_0(X)
        elif idx == [1]:
            output = f_1(X)

        return output
