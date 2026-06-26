"""
This module contains functions and classes to train neural importance sampling networks and
evaluate the integration and sampling performance.
"""

from .buffer import Buffer
from .channel_grouping import ChannelData, ChannelGroup, ChannelGrouping
from .distribution import Distribution
from .integrand import Integrand
from .integrator import Integrator, SampleBatch, TrainingStatus
from .interface import (
    IntegrandDistribution,
    IntegrandFunction,
    build_madnis_integrand,
)
from .losses import (
    kl_divergence,
    kl_divergence_softclip,
    multi_channel_loss,
    stratified_variance,
    stratified_variance_softclip,
    variance,
)

__all__ = [
    "Integrator",
    "TrainingStatus",
    "SampleBatch",
    "Integrand",
    "Buffer",
    "multi_channel_loss",
    "stratified_variance",
    "stratified_variance_softclip",
    "variance",
    "kl_divergence",
    "kl_divergence_softclip",
    "ChannelGroup",
    "ChannelData",
    "ChannelGrouping",
    "Distribution",
    "IntegrandDistribution",
    "IntegrandFunction",
    "build_madnis_integrand",
]
