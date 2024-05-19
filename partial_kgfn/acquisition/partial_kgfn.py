#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
An implementation of the knowledge gradient acquisition function for function networks 
with PARTIAL evaluations using the discretization approach.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.logging import logger
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.sampling import ListSampler, MCSampler, SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class PartialKnowledgeGradientFN(MCAcquisitionFunction):
    r"""MC-based knowledge gradient acquisition function with PARTIAL function network evaluations.

    This computes the knowledge gradient acquisition for function networks with partial 
    evaluations using fantasies for the outer expectation and MC-sampling for
    the inner expectation calculated only on a discrete set of `X_fantasies` candidates.

    Candidates for `X_fantasies` for fantasy models need to be provided.
    """
    def __init__(
        self,
        model: Model,
        d: int,
        q: int,
        node_dim: int,
        problem_bounds: Tensor,
        X_fantasies_candidates: Optional[Tensor] = None,
        X_evaluation_mask: Optional[Tensor] = None,
        num_fantasies: Optional[int] = None,
        sampler: Optional[MCSampler] = None,
        use_posterior_mean: bool = True,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        inner_sampler: Optional[MCSampler] = None,
        current_value: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""Discrete Knowledge Gradient for function network.

        Args:
            model: A fitted GP network model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            d: The original problem dimension.
            q: The number of points to be fantasized.
            node_dim: An input dimension at the node of interest.
            problem_bounds: A 2xd tensor of bounds for each dimension.
            X_fantasies_candidates: A n_grid x d tensor of discretized candidates for inner optimization.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The MC objective under which the samples are evaluated.
            posterior_transform: An optional PosteriorTransform. If given, this
                transforms the posterior before evaluation. If `objective is None`,
                then the analytic posterior mean of the transformed posterior is
                used. If `objective` is given, the `inner_sampler` is used to draw
                samples from the transformed posterior, which are then evaluated under
                the `objective`.
            inner_sampler: The sampler used for inner sampling.
            X_evaluation_mask: A `n_nodes`-dim binary tensor with only one 1's element
                indicating the node index which the q points will be fantasized.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
                
        Returns:
            None.
        """
        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            sampler = [
                SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
                for _ in range(sum(X_evaluation_mask))
            ]
        elif num_fantasies is not None:
            sample_shape = (
                sampler.samplers[0].sample_shape
                if isinstance(sampler, ListSampler)
                else sampler.sample_shape
            )
            if sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler[0].sample_shape[0]
        super().__init__(model=model, objective=objective)
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        elif objective is not None and not isinstance(
            objective, MCAcquisitionObjective
        ):
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
        if objective is None and model.num_outputs != 1:
            if posterior_transform is None:
                raise UnsupportedError(
                    "Must specify an objective or a posterior transform when using "
                    "a multi-output model."
                )
            elif not posterior_transform.scalarize:
                raise UnsupportedError(
                    "If using a multi-output model without an objective, "
                    "posterior_transform must scalarize the output."
                )
        if X_fantasies_candidates is None:
            self.X_fantasies_candidates = draw_sobol_samples(
                bounds=problem_bounds,
                n=50 * d,
                q=1,
            ).squeeze(-2)
        else:
            self.X_fantasies_candidates = X_fantasies_candidates
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.inner_sampler = inner_sampler
        self.num_fantasies = num_fantasies
        self.current_value = current_value
        self.use_posterior_mean = use_posterior_mean
        self.sampler = sampler
        self.X_evaluation_mask = X_evaluation_mask
        self.d = d
        self.q = q
        self.node_dim = node_dim

    @t_batch_mode_transform()
    def forward(self, X: Tensor, verbose=False) -> Tensor:
        r"""Evaluate knowledge gradient for function network with partial evaluations 
        on the candidate set `X`.

        Args:
            X: A `b x q x d_k`-dim Tensor with `b` t-batches of `q` design points to be evaluated
                at a selected node k. Here, `d_k` is the dimension of `z`, the input for node `k`.
            Note: Current version only allows `q=1`.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape x 1`.
        """
        if sum(self.X_evaluation_mask) > 1:
            k = (self.X_evaluation_mask == 1).nonzero().squeeze()[0]
        else:
            k = (self.X_evaluation_mask == 1).nonzero().squeeze()
        if isinstance(self.model.node_GPs[k], GenericDeterministicModel):
            fantasy_model = self.model
        else:
            fantasy_model = self.model.fantasize(
                X=[X] * sum(self.X_evaluation_mask),
                samplers=self.sampler,
                observation_noise=True,
                evaluation_mask=self.X_evaluation_mask,
            )
        fantasy_post_at_X = fantasy_model.posterior(
            X=self.X_fantasies_candidates
        ) 
        values = self.inner_sampler(
            fantasy_post_at_X
        ) 
        values = values.mean(dim=0)
        if self.objective is not None:
            values = self.objective(values)
        else:
            values = values[..., -1]
        values, max_points = torch.max(
            values, dim=-1
        )
        if verbose:
            print(f"Points attain max value: {max_points}")
        values = values.mean(dim=0)
        if self.current_value is not None:
            values = values - self.current_value
        logger.debug(f"X: {X}")
        logger.debug(f"Acqf value {values}")
        return values

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        return 1
