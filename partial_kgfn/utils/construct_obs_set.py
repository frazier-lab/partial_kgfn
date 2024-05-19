#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Reconstruct the dataset as the input for the GP network model.
"""


from typing import List, Optional, Tuple

import torch
from torch import Tensor


def construct_obs_set(
    X: Tensor,
    Y: Tensor,
    parent_nodes: List[List[Optional[int]]],
    active_input_indices: List[List[Optional[int]]],
) -> Tuple[List[Tensor], List[Tensor]]:
    """Resconstuct the sets of training inputs and outputs in the designed format to fit the GP network model.

    Args:
        X (Tensor): A 'batch_shape x d' Tensor of training inputs.
        Y (Tensor): A 'batch_shape x 1' Tensor of outputs.
        parent_nodes (List[List[Optional[int]]]): A list of lists of parent node indices for each node
        active_input_indices (List[List[Optional[int]]]):  A list of lists of indices of the active inputs for each node.

    Returns:
        A two-element tuple containing:
            - A list of tensors of training inputs for individual node GP.
            - A list of tensors of outputs for inidividual node GP
    """
    num_nodes = len(parent_nodes)
    train_X = []
    train_Y = []
    for idx in range(num_nodes):
        if len(parent_nodes[idx]) != 0 and len(active_input_indices[idx]) != 0:
            if X[..., active_input_indices[idx]].ndim == 1:
                active_part = X[..., active_input_indices[idx]].unsqueeze(dim=-1)
            else:
                active_part = X[..., active_input_indices[idx]]
            if Y[..., parent_nodes[idx]].ndim == 1:
                parent_part = Y[..., parent_nodes[idx]].unsqueeze(dim=-1)
            else:
                parent_part = Y[..., parent_nodes[idx]]
            if parent_part.ndim > active_part.ndim:
                shape_list = [1 for _ in range(parent_part.ndim)]
                shape_list[0] = parent_part.shape[0]
                shape_rep = torch.Size(shape_list)
                active_part = active_part.unsqueeze(0).repeat(shape_rep)
            train_X_temp = torch.cat((active_part, parent_part), dim=-1)
        elif len(parent_nodes[idx]) != 0 and len(active_input_indices[idx]) == 0:
            if Y[..., parent_nodes[idx]].ndim == 1:
                train_X_temp = Y[..., parent_nodes[idx]].unsqueeze(dim=-1)
            else:
                train_X_temp = Y[..., parent_nodes[idx]]
        elif len(parent_nodes[idx]) == 0 and len(active_input_indices[idx]) != 0:
            if X[..., active_input_indices[idx]].ndim == 1:
                active_part = X[..., active_input_indices[idx]].unsqueeze(dim=-1)
            else:
                active_part = X[..., active_input_indices[idx]]
            train_X_temp = active_part
        train_X.append(train_X_temp)
        train_Y.append(Y[..., idx].unsqueeze(dim=-1))
    return train_X, train_Y
