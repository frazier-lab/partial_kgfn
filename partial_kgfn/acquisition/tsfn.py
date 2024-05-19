#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
An implementation of the Thompson sampling acquisition function for function networks.
"""
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import Model
from botorch.utils.gp_sampling import get_gp_samples
from torch import Tensor


class ThompsonSamplingFN(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        objective: Optional[GenericMCObjective] = None,
    ) -> None:
        r"""The Thompson sampling acquisition function for function networks.

        Args:
            model: A fitted GP network model.
            objective: An objective for the problem.
            
        Returns:
            None.
        """
        super(AcquisitionFunction, self).__init__()
        self.model = model
        self.GP_samples = [None for i in range(model.n_nodes)]
        self.objective = objective
        for i in range(model.n_nodes):
            self.GP_samples[i] = get_gp_samples(
                model=model.node_GPs[i], num_outputs=1, n_samples=1
            )

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition value for function network on the candidate set `X`.

        Args:
            X: input tensor of shape `batch_shape x q x d` to evaluate the function network at.
        
        Returns:
            A  `batch_shape`-dim tensor of acquisition values.
        """
        nodes_samples = torch.empty(
            (X.shape[:-1] + torch.Size([self.model.n_nodes])), dtype=torch.double
        )
        nodes_samples_available = [False for k in range(self.model.n_nodes)]
        for k in self.model.root_nodes:
            if self.model.active_input_indices is not None:
                X_node_k = X[..., self.model.active_input_indices[k]]
            else:
                X_node_k = X
            nodes_samples[..., [k]] = self.GP_samples[k].posterior(X_node_k).mean
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.model.n_nodes):
                parent_nodes = self.model.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all(
                    [nodes_samples_available[j] for j in parent_nodes]
                ):
                    parent_nodes_samples = nodes_samples[..., parent_nodes]
                    X_node_k = X[..., self.model.active_input_indices[k]]
                    X_node_k = torch.cat([X_node_k, parent_nodes_samples], -1)
                    nodes_samples[..., [k]] = (
                        self.GP_samples[k].posterior(X_node_k).mean
                    )
                    nodes_samples_available[k] = True
        output = self.objective(nodes_samples)
        if output.ndim != 1:
            output = output.squeeze(-1)
        return output
