"""Implementation of Conditional Glow."""

import torch
import torch.nn as nn
from torch.nn import functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import BatchNorm, ActNorm
from nflows.utils import torchutils

class ConditionalGlow(Flow):
    """ A version of Conditional Glow for 1-dim inputs.

    Reference:
    > TODO
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        activation=F.relu,
        dropout_probability=0.5,
        context_features=None,
        batch_norm_within_layers=True,
    ):

        coupling_constructor = AdditiveCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                context_features=context_features,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            layers.append(ActNorm(features=features))
            layers.append(LULinear(features=features))
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            mask *= -1
            layers.append(transform)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features])
        )

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        return log_prob + logabsdet, noise

    def sample_and_log_prob(self, num_samples, noise=None, context=None, ignore_context=False):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        if noise is None:
            noise, log_prob = self._distribution.sample_and_log_prob(
                num_samples, context=embedded_context
            )
        else:
            batch_size = noise.shape[0]
            num_samples = noise.shape[1]
            log_prob = self._distribution.log_prob(
                noise.reshape(batch_size * num_samples, -1), context=embedded_context
            )
            log_prob = log_prob.reshape(batch_size, num_samples)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context if not ignore_context else None)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet, noise