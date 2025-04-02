#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
An implementation of the knowledge gradient for function networks with FULL evaluations 
using discretization approach.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from fast_pkgfn.utils.construct_obs_set import construct_obs_set

TAcqfArgConstructor = Callable[[Model, Tensor], Dict[str, Any]]
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class FullKnowledgeGradientFN(MCAcquisitionFunction):
    r"""MC-based knowledge gradient acquisition function with FULL function network evaluations."""

    def __init__(
        self,
        d: int,
        model: Model,
        problem_bounds: Tensor,
        batch_sizes: List[int] = None,
        X_fantasies_candidates: Optional[Tensor] = None,
        num_fantasies: Optional[List[int]] = None,
        samplers: Optional[List[MCSampler]] = None,
        current_value: Optional[Tensor] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        inner_sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Knowledge gradient acquisition function for function networks with FULL evaluations and discretization.

        Args:
            d: the problem dimension.
            model: A fitted `GaussianProcessNetwork` model.
            problem_bounds: A 2xd tensor of bounds for each dimension.
            batch_sizes: A list of `n_sub` integers indicating. All are 1's.
            X_fantasies_candidates: A n_disc x d tensor of discrete candidates for inner optimization.
            num_fantasies: A list of `n_sub` integers, i.e. `[f_s1, ..., f_s(n_sub)]' containing the number 
                of fantasy values at a point in each node_group in a subtree.
            samplers: A list of MCSampler objects to be used for sampling fantasy values in each node.
                    **Ignore if num_fantasies is provided.**
            current_value: The maximum of posterior mean of the current `GaussianProcessNetwork` model before           
                fantasizing, i.e., KG constant.
            objective: The MC-objective under which the output is evaluated.
            posterior_transform: An optional PosteriorTransform. If given, this transforms the posterior 
                before evaluation.
                If `objective is None', then the output of the transformed posterior is used. If `objective' is given,
                the `inner_sampler' is used to draw samples from the transformed posterior, which are then 
                evaluated under the `objective'.
            inner_samples: A SobolQMCNormalSampler for SAA to compute the KG value.
        
        Returns:
            None.
        """
        if not isinstance(objective, MCAcquisitionObjective):
            if posterior_transform is None:
                posterior_transform = self._deprecate_acqf_objective(
                    posterior_transform=posterior_transform,
                    objective=objective,
                )
                objective = None
            else:
                raise RuntimeError(
                    "Got both a non-MC objective (DEPRECATED) and a posterior "
                    "transform. Use only a posterior transform instead."
                )
        super(MCAcquisitionFunction, self).__init__(model=model)

        self.d = d

        if batch_sizes is not None:
            self.batch_sizes = batch_sizes
        else:
            self.batch_sizes = [1 for _ in range(model.n_nodes)]
        if not ((num_fantasies is None) ^ (samplers is None)):
            raise UnsupportedError(
                "fullKGFN requires exactly one of `num_fantasies` or "
                "`samplers` as arguments."
            )
        if samplers is None:
            samplers = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        else:
            num_fantasies = samplers.sample_shape[0]
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        self.num_fantasies = num_fantasies
        self.objective = objective
        self.model = model
        self.posterior_transform = posterior_transform
        self.samplers = samplers
        self.inner_samplers = inner_sampler
        self.current_value = current_value
        if X_fantasies_candidates is None:
            self.X_fantasies_candidates = draw_sobol_samples(
                bounds=problem_bounds,
                n=50 * d,
                q=1,
            ).squeeze(-2)
        else:
            self.X_fantasies_candidates = X_fantasies_candidates

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the knowledge gradient acquisition function for function networks with FULL 
            evaluations on the candidate set X.

        Args:
            X: A `batch_shape x 1 x d`-dim tensor of design points to be evaluated.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape x 1`.
        """
        X_fantasies = self.X_fantasies_candidates
        fantasy_model = self.model
        posterior_at_X = self.model.posterior(X=X)
        network_sample = self.samplers(posterior_at_X)
        fan_X, fan_Y = construct_obs_set(
            X=X,
            Y=network_sample,
            parent_nodes=self.model.dag.parent_nodes,
            active_input_indices=self.model.active_input_indices,
        )
        fantasy_model = fantasy_model.condition_on_observations(X=fan_X, Y=fan_Y)
        fantasy_post_at_X = fantasy_model.posterior(X=X_fantasies)
        values = self.inner_samplers(fantasy_post_at_X)
        values = values.mean(dim=0)
        if self.objective is not None:
            values = self.objective(values)
        else:
            values = values[
                ..., -1
            ]
        values, _ = torch.max(values, dim=-1)
        values = values.mean(dim=0)
        if self.current_value is not None:
            values = values - self.current_value
        return values
