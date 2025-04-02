#! /usr/bin/env python3

r"""
Gaussian Process Network.
Modified from https://github.com/RaulAstudillo06/BOFN/blob/main/bofn/models/gp_network.py 
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from botorch import fit_gpytorch_model
from botorch.acquisition.objective import PosteriorTransform
from botorch.logging import logger
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from botorch.posteriors import TorchPosterior
from botorch.sampling import MCSampler
from botorch.test_functions import SyntheticTestFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from fast_pkgfn.models.dag import DAG
from fast_pkgfn.test_functions.manufacter_gp import ManufacturingGPNetwork

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class GaussianProcessNetwork(Model):
    r"""A model class represented by a function network of Gaussian processes."""

    def __init__(
        self,
        train_X: List[Tensor],
        train_Y: List[Tensor],
        dag: DAG,
        active_input_indices: List[List[int]],
        node_groups: Optional[List[List[int]]] = None,
        node_dims: Optional[List[List[int]]] = None,
        train_Yvar: Optional[List[Tensor]] = None,
        node_GPs: Optional[List[Model]] = None,
        problem: Optional[SyntheticTestFunction] = None,
        noisy: Optional[bool] = False,
    ) -> None:
        r"""Initiate a Gaussian process network model.

        Args:
            train_X: A length-`n_nodes` list of training input data, where the `k`-th
                element is the training input for node `k`.
            train_Y: A length-`n_nodes` list of training output data, where the `k`-th
                element is the training output for node `k`.
            dag: An class object of directed acyclic graph (DAG) representing a
                function network.
            active_input_indices: A length-`n-nodes` list of active input indices, where
                the `k`-th element is the indices of active inputs of `X` for node `k`.
            node_groups: A list of list of integers representing groups of nodes in function networks. 
                Nodes in the same group must have same types of inputs and must be evaluated simultaneously.
            node_dims: A list of integers representing inputs' dimension of each node
            train_Yvar: A length-`n_nodes` list of training output variance, where the
                `k`-th element is the trainint output for node `k` (Optional).
            node_GPs: A length-`n-nodes` list of GPs representing the function nodes
                in the function network (Optional).
            problem: A problem class representing the test case being considered

        Returns:
            None
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.node_dims = node_dims
        self.root_nodes = dag.get_root_nodes()
        self.active_input_indices = active_input_indices
        self.noisy = noisy
        if node_groups is None:
            self.node_groups = [[x] for x in range(self.n_nodes)]
        else:
            self.node_groups = node_groups
        if train_Yvar is not None:
            self.train_Yvar = train_Yvar
        else:
            self.train_Yvar = [
                torch.ones_like(self.train_Y[k]) * 1e-6 for k in range(self.n_nodes)
            ]  
        if isinstance(problem, ManufacturingGPNetwork):
            self.modify_prior = True
        else:
            self.modify_prior = False
        if node_GPs is not None:
            self.node_GPs = node_GPs
        else:
            self.node_GPs = [None for k in range(self.n_nodes)]

            for k in range(self.n_nodes):
                self.node_GPs[k] = initialize_GP(
                    train_X=self.train_X[k],
                    train_Y=self.train_Y[k],
                    train_Yvar=self.train_Yvar[k],
                    modified_prior=self.modify_prior,
                    noisy=self.noisy,
                )

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model.

        Although the number of nodes in the network can be more than one, the number of
        outputs, i.e. the dimension of the objective is set to one.
        """
        return 1

    @property
    def num_nodes(self) -> int:
        r"""The number of nodes in the function network."""
        return self.n_nodes

    @property
    def batch_shape(self) -> torch.Size:
        """compute the batch shape of the GaussianProcessNetwork model."""
        # gp_batch_shapes = [gp.batch_shape for gp in self.node_GPs]
        # idx = torch.argmax(torch.Tensor([len(shape) for shape in gp_batch_shapes]))
        # gp_batch_shape = gp_batch_shapes[idx]
        gp_batch_shape = torch.broadcast_shapes(
            *[gp.batch_shape for gp in self.node_GPs]
        )
        return gp_batch_shape

    def posterior(
        self,
        X: Tensor,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> MultivariateNormalNetwork:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
            TODO: implement posterior_transform

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        return MultivariateNormalNetwork(
            self.node_GPs, self.dag, X, self.active_input_indices
        )

    # def forward(self, X: Tensor) -> MultivariateNormalNetwork:
    #     return MultivariateNormalNetwork(
    #         node_GPs=self.node_GPs,
    #         dag=self.dag,
    #         X=X,
    #         indices_X=self.active_input_indices,
    #     )

    def condition_on_observations(
        self,
        X: List[Tensor],
        Y: List[Optional[Tensor]],
        Yvar: Optional[Optional[List[Tensor]]] = None,
        **kwargs: Any,
    ) -> Model:
        r"""Condition the model on new observations.

        Args:
            X: A length-`n_nodes` list of tensors containing new designs at each node to be
                conditioned on. For the node NOT to be updated, X at the index will be None.
            Y: A length-`n_nodes`list of tensors containing new outputs corresponding to new
                designs at each node to be conditioned on. For the node NOT to be updated,
                Y at the index will be None.
            Yvar: A length-`n_nodes` list of tensors containing variance of new outputs
                corresponding to new designs at each node to be conditioned on
            Note that: evaluation_mask can be extracted by locating all indices with None values.
            TODO: deal with kwargs

        Returns:
            An updated network `Model` object of the same type, representing the original model
            conditioned on the new observations.
        """
        fantasy_models = [None] * self.n_nodes

        for k in range(self.n_nodes):
            if Y[k] is not None:
                if Yvar is not None and Yvar[k] is None:
                    raise ValueError(f"Yvar for node {k} must be provided.")

                Yvar_node_k = (
                    Yvar[k]
                    if Yvar is not None
                    else torch.ones(Y[k].shape[1:]).to(Y[k]) * 1e-6  
                )
                fantasy_model = self.node_GPs[k].condition_on_observations(
                    X=self.node_GPs[k].transform_inputs(X[k]),
                    Y=Y[k],
                    noise=Yvar_node_k,
                )
                fantasy_models[k] = fantasy_model

            else:
                fantasy_models[k] = self.node_GPs[k]

        return GaussianProcessNetwork(
            dag=self.dag,
            train_X=self.train_X,
            train_Y=self.train_Y,
            train_Yvar=self.train_Yvar,
            active_input_indices=self.active_input_indices,
            node_GPs=fantasy_models,
        )
        # note that we do not update the train_X and train_Y here,
        # since we do not need to access them in the fantasy model.
        # TODO: clean up `train_X` and `train_Y` in the GaussianProcessNetwork model.

    def fantasize(
        self,
        X: List[Tensor],
        samplers: List[MCSampler],
        evaluation_mask: Tensor,
        **kwargs: Any,
    ) -> Model:
        r"""Construct the fantasy model.

        Args:
            X: A list of length `node_update` Tensors, where `node_update` is the number of
                function nodes in a network to be fantasized, i.e. the number of ones
                in evaluation_mask. Each of which has dimension `n_data_node x d_node`,
                where `n_data_node` is a number of data points to be fantasized and `d_node`
                is the dimension of the feature space at a particular node.
            samplers: A list of MC samplers of legnth `node_update`.
            evaluation_mask: A binary list of length `n_node` where 1 means that a node is selected
                to be fantasized.
            TODO: deal with kwargs in the fantasize() function.

        Returns:
            An updated GP network.
        """
        if len(X) != sum(evaluation_mask):
            raise ValueError(
                f"Length of X does not math the number of nodes to be fantasized"
            )

        X_for_condition = [None] * self.n_nodes
        Y_for_condition = [None] * self.n_nodes
        list_idx = 0
        for k in range(self.n_nodes):
            if evaluation_mask[k] == 0:
                continue

            X_node_k = X[list_idx]
            X_for_condition[k] = X_node_k
            posterior_X_node_k = self.node_GPs[k].posterior(X_node_k)
            Y_node_k = samplers[list_idx](posterior_X_node_k)
            Y_for_condition[k] = Y_node_k
            list_idx = list_idx + 1

        # update at once using the `condition_on_observation` function
        fantasy_models = self.condition_on_observations(
            X=X_for_condition, Y=Y_for_condition
        )
        return fantasy_models


class MultivariateNormalNetwork(TorchPosterior):
    r"""Posterior of the Gaussian process network."""

    def __init__(
        self,
        node_GPs: List[Model],
        dag: DAG,
        X: Tensor,
        indices_X: List[List[int]] = None,
    ) -> None:
        r"""Initiate the posterior for a Gaussian process network.

        Args:
            node_GPs: A length-`n-nodes` list of GPs representing the function nodes
                in the function network.
            dag: A class object of directed acyclic graph (DAG) representing a
                function network.
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            indices_X: A length-`n-nodes` list of active input indices, where
                the `k`-th element is the indices of active inputs of `X` for node `k`.

        Returns:
            None
        """
        self.node_GPs = node_GPs
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.X = X
        self.active_input_indices = indices_X

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return tkwargs["device"]

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return tkwargs["dtype"]

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = [self.X.shape[-2], self.n_nodes]
        shape = torch.Size(shape)
        return self.batch_shape + shape

    @property
    def batch_shape(self) -> torch.Size:
        """compute the batch shape of the GaussianProcessNetwork posterior."""
        # gp_batch_shapes = [gp.batch_shape for gp in self.node_GPs]
        # idx = torch.argmax(torch.Tensor([len(shape) for shape in gp_batch_shapes]))
        # gp_batch_shape = gp_batch_shapes[idx]
        gp_batch_shape = torch.broadcast_shapes(
            *[gp.batch_shape for gp in self.node_GPs]
        )
        X_batch_shape = self.X.shape[:-2]
        return torch.broadcast_shapes(gp_batch_shape, X_batch_shape)

    # base_sample_shape now returns torch.Size([n_f, q, n_nodes])
    @property
    def base_sample_shape(self) -> torch.Size:
        """Compute the base sample shape of the GaussianProcessNetwork posterior."""
        return self.event_shape

    # _extended_shape return torch.Size([n_SAA, n_f, q, n_nodes])
    @property
    def _extended_shape(self, sample_shape: torch.Size) -> torch.Size:
        return sample_shape + self.base_sample_shape

    @property
    def batch_range(self) -> Tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return (0, -2)

    def rsample(self):
        return None  # TODO

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size(),
        base_samples: Tensor,
    ) -> Tensor:
        """Sample from the posterior.

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape.
            base_samples: A Tensor of `N(0, I)` base samples of shape
                `sample_shape x base_sample_shape`, typically obtained from
                an `MCSampler`. This is used for deterministic optimization.

        Returns:
            A `sample_shape x self.event_shape`-dim tensor of samples drawn from the posterior.
        """
        nodes_samples = torch.empty(
            base_samples.shape
        )  # sample_shape x batch_shape x q x m
        nodes_samples = nodes_samples.to(self.device).to(self.dtype)
        nodes_samples_available = [False for k in range(self.n_nodes)]
        batch_shape = base_samples.shape[len(sample_shape) : -2]

        # We make a replication of self.X so that it has dimension torch.Size([b_2 x b_1, q, d])
        # If the GP network model does not have batch shape, skip this step
        if len(batch_shape) > 0:
            self.X = torch.broadcast_to(self.X, batch_shape + self.X.shape[-2:])
            # self.X = self.X[(None,) * len(batch_shape)]
            # self.X = self.X.repeat(*list(batch_shape), 1, 1)

        for k in self.root_nodes:
            if self.active_input_indices is not None:
                X_node_k = self.X[..., self.active_input_indices[k]]
            else:
                X_node_k = self.X  # batch_shape x q x d
            multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
            # two cases:
            # case 1: node_GP has no fantasy dimension (i.e., input is q x d), then X_node_k is shape nf x q x d, posterior is nf x q x 1
            # case 2: node_GP has fantasy dimension (i.e., input is n_f x q x d), then X_node_k is shape nf x q x d, posterior is nf x q x 1
            nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(
                sample_shape, base_samples[..., [k]]  # sample_shape x n_f x q x 1
            )[..., 0]
            # base_sample: sample_shape x batch_shape x q x m
            # left: sample_shape x batch_shape x q
            # right: sample_shape x batch_shape x q

            # nodes_samples[...,k] has dimension torch.Size([n_SAA, n_f, q])
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.n_nodes):
                parent_nodes = self.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all(
                    [nodes_samples_available[j] for j in parent_nodes]
                ):
                    parent_nodes_samples = nodes_samples[
                        ..., parent_nodes
                    ]  # n_SAA x nf x q x n_parents
                    X_node_k = self.X[
                        ..., self.active_input_indices[k]
                    ]  # n_f x q x d_k_active
                    aux_shape = [sample_shape[0]] + [
                        1
                    ] * X_node_k.ndim  # n_SAA x 1 x 1 x 1
                    X_node_k = X_node_k.unsqueeze(0).repeat(
                        *aux_shape
                    )  # n_SAA x n_f x q x d_k_active
                    X_node_k = torch.cat(
                        [X_node_k, parent_nodes_samples], -1
                    )  # n_SAA x n_f x q x d_aug where d_aug = d_k_active + n_parents_node_k
                    multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
                    # two cases:
                    # case 1: node_GP has no fantasy dimension (i.e., input is q x d_aug), then X_node_k is shape n_SAA x nf x q x d_aug, posterior is n_SAA x nf x q x 1
                    # case 2: node_GP has fantasy dimension (i.e., input is n_f x q x d_aug), then X_node_k is shape n_SAA x nf x q x d_aug, posterior is n_SAA x nf x q x 1
                    # if base_samples is not None:
                    nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(
                        sample_shape=torch.Size([1]),
                        base_samples=base_samples[..., [k]].unsqueeze(dim=0),
                        # base_samples=base_samples[..., [k]],
                    )[0, ..., 0]
                    # print(nodes_samples_k)
                    # note: base_samples: currently n_SAA x n_f x q x 1 but we want 1 x n_SAA x n_f x q x 1
                    # result: 1 x n_SAA x nf x q x 1 -> n_SAA x nf x q
                    # else:
                    #     nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(
                    #         sample_shape=torch.Size([1])
                    #     )[0, ..., 0]
                    nodes_samples_available[k] = True
        return nodes_samples


def fit_gp_network(model: GaussianProcessNetwork, idx: Optional[int] = None) -> None:
    r"""Fit GP network model.

    Args:
        model: A GaussianProcessNetwork object.
        idx: The index of the node GP to be fixed. If None, all node GPs will be refitted.

    Returns:
        None.
    """
    if idx is None:
        for k in range(model.n_nodes):
            node_mll = ExactMarginalLogLikelihood(
                model.node_GPs[k].likelihood,
                model.node_GPs[k],
            )
            fit_gpytorch_model(node_mll)
    else:
        node_mll = ExactMarginalLogLikelihood(
            model.node_GPs[idx].likelihood,
            model.node_GPs[idx],
        )
        fit_gpytorch_model(node_mll)


def initialize_GP(
    train_X, train_Y, modified_prior: bool, train_Yvar=None, noisy=False
) -> Model:
    r"""Initialize a GP model.

    Args:
        train_X: A `batch_shape x n x d` tensor of training inputs.
        train_Y: A `batch_shape x n x m` tensor of training targets.
        train_Yvar: Optional `batch_shape x n x m` tensor of training noise variances.

    Returns:
        A GP model
    """
    dim, batch_shape = train_X.shape[-1], train_X.shape[:-2]
    output_dim = train_Y.shape[-1]

    if train_Yvar is None:
        train_Yvar = torch.ones_like(train_Y) * 1e-6  
    if not noisy:
        model = FixedNoiseGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=Standardize(m=output_dim, batch_shape=batch_shape),
            input_transform=Normalize(d=dim),
        )
        if modified_prior:
            model.covar_module.base_kernel.lengthscale_prior.concentration = (
                torch.Tensor([5]).to(torch.double)
            )
            model.covar_module.base_kernel.lengthscale_prior.rate = torch.Tensor(
                [2]
            ).to(torch.double)
    else:
        logger.info(f"Consider noisy observations: SingleTaskGP is used.")
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=output_dim, batch_shape=batch_shape),
            input_transform=Normalize(d=dim),
        )
        if modified_prior:
            model.covar_module.base_kernel.lengthscale_prior.concentration = (
                torch.Tensor([5]).to(torch.double)
            )
            model.covar_module.base_kernel.lengthscale_prior.rate = torch.Tensor(
                [2]
            ).to(torch.double)
    return model
