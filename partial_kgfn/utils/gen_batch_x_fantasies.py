#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Generate batch of X_fantasies candidate for knowledge gradient discrete-type acquisition functions
"""
from typing import Callable, List, Optional, Union

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import Model
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class GenbatchXFantasiesFN(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        num_samples: Optional[int] = 5,
        objective: Optional[GenericMCObjective] = None,
    ) -> None:
        r"""Thompson Sampling Acquisition Function.

        Args:
            model: A model class approximating a function network
            num_samples: number of sampled functions at each node from its posterior distributions.
        """
        super(AcquisitionFunction, self).__init__()
        self.model = model
        self.GP_samples = [None for i in range(model.n_nodes)]
        self.objective = objective
        self.num_samples = num_samples
        for i in range(model.n_nodes):
            self.GP_samples[i] = get_gp_samples(
                model=model.node_GPs[i], num_outputs=1, n_samples=num_samples
            )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Sample from the posterior.

        Args:
            X: input tensor "batch x q x d" to evaluate the function network at.
        Returns:
            A tensor "batch"-dim of final node function values
        """
        nodes_samples = torch.empty(
            (
                torch.Size([self.num_samples])
                + X.shape[:-1]
                + torch.Size([self.model.n_nodes])
            ),
            dtype=torch.double,
        )
        nodes_samples_available = [False for _ in range(self.model.n_nodes)]
        for k in self.model.root_nodes:
            if self.model.active_input_indices is not None:
                X_node_k = X[..., self.model.active_input_indices[k]]
            else:
                X_node_k = X
            nodes_samples[..., [k]] = torch.cat(
                [
                    self.GP_samples[k]
                    .posterior(X_node_k[[i], ...])
                    .mean.transpose(1, -1)
                    for i in range(X_node_k.shape[0])
                ],
                dim=1,
            ).unsqueeze(-1)
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.model.n_nodes):
                parent_nodes = self.model.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all(
                    [nodes_samples_available[j] for j in parent_nodes]
                ):
                    parent_nodes_samples = nodes_samples[..., parent_nodes]
                    X_node_k = X[..., self.model.active_input_indices[k]].unsqueeze(0)
                    repeat_size = (self.num_samples,) + torch.Size(
                        [1] * len(X_node_k.shape[1:])
                    )
                    X_node_k = X_node_k.repeat(repeat_size)
                    X_node_k = torch.cat([X_node_k, parent_nodes_samples], -1)
                    nodes_samples[..., [k]] = torch.cat(
                        [
                            self.GP_samples[k]
                            .posterior(X_node_k[:, i, ...])
                            .mean.transpose(1, -1)
                            for i in range(X_node_k.shape[1])
                        ],
                        dim=1,
                    ).unsqueeze(-1)
                    nodes_samples_available[k] = True
        output = self.objective(nodes_samples)
        if output.ndim != 1:
            output = output.squeeze(-1)
        output, _ = torch.max(output, dim=-1)
        output = torch.mean(output, dim=0)
        return output