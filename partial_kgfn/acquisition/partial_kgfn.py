#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
An implementation of the knowledge gradient for function networks with partial evaluations 
using discretization approach.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from botorch.acquisition.cost_aware import CostAwareUtility
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
    r"""Batch Discrete Knowledge Gradient for Function Network.

    This computes the batch Discrete Knowledge Gradient for function network
    using fantasies for the outer expectation and MC-sampling for
    the inner expectation calculated only on discretized X_fantasies candidates.

    Discretized candidates for X_fantasies for fantasy models have been provided.
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
        cost_aware_utility: Optional[CostAwareUtility] = None,
        cost_sampler: Optional[MCSampler] = None,
        **kwargs: Any,
    ) -> None:
        r"""Discrete Knowledge Gradient for function network.

        Args:
            model: A fitted GP network model. Must support fantasizing.
            d: The original problem dimension.
            q: The number of points to be fantasized.
            node_dim: An input dimension at the node of interest.
            problem_bounds: A 2xd tensor of bounds for each dimension.
            X_fantasies_candidates: A n_grid x d tensor of discretized candidates for inner optimization.
            X_evaluation_mask: A `n_nodes`-dim binary tensor with only one 1's element
                indicating the node index which the q points will be fantasized.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
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
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
        """

        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
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
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        elif objective is not None and not isinstance(
            objective, MCAcquisitionObjective
        ):
            # TODO: clean this up after removing AcquisitionObjective.
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
            # If a set of discretized X_fantasies_candidates is not provided, randomly generate one.
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
        self.cost_aware_utility = cost_aware_utility
        self.use_posterior_mean = use_posterior_mean
        self.cost_sampler = cost_sampler
        self.sampler = sampler
        self.X_evaluation_mask = X_evaluation_mask
        self.d = d
        self.q = q
        self.node_dim = node_dim

    @t_batch_mode_transform()
    def forward(self, X: Tensor, verbose=False) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x q x d_k` Tensor with `b` t-batches of
                `q` design points to be fantasized at a selected node k.
                Note: Current version only allows q=1.
                `d_k` is the dimension of z, the input for node k

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KGFN value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model from the set of X_fantasies_candidates previously provided.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        # We can do the following lines as now we consider nodes in a group that take the same input
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
            )  # n_f x b
        fantasy_post_at_X = fantasy_model.posterior(
            X=self.X_fantasies_candidates
        )  # num_fantasies x b x 1 x d
        values = self.inner_sampler(
            fantasy_post_at_X
        )  # n_SAA x num_fantasies x b x n_grid x n_nodes, where n_grid is a number of discretized X_fantasies candidates
        values = values.mean(dim=0)
        if self.objective is not None:
            values = self.objective(values)
        else:
            values = values[..., -1]  # values now has shape num_fantasies x b x n_grid
        values, max_points = torch.max(
            values, dim=-1
        )  # taking maximum of posterior mean for each fantasy model out of n_grid candidate points
        if verbose:
            print(f"Points attain max value: {max_points}")
        # values now has shape num_fantasies x b
        values = values.mean(dim=0)  # averaging over fantasies points
        # values now has shape b
        if self.current_value is not None:
            values = values - self.current_value
        logger.debug(f"This is X: {X}")
        logger.debug(f"This is value {values}")
        return values

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        # TODO: to update this function
        # return q + self.num_fantasies
        # it is always one!
        return 1
