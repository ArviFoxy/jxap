"""Common types for audio plugins."""

from __future__ import annotations

import abc
from typing import Mapping, Sequence, Protocol, Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Float

# Audio buffer type.
# Samples are ordered so that:
#   - buffer[0] is the earliest played sample.
#   - buffer[1] is the second earliest played sample.
#   - buffer[-1] is the latest.
Buffer = Float[Array, "BufferSize"]

# Re-exported for convenience.
Module = eqx.Module

# Type of the plugin state.
PluginState = TypeVar("PluginState", bound=Module)


class Plugin(Protocol, Generic[PluginState]):
    """Audio plugin with state `PluginState`."""

    @property
    @abc.abstractmethod
    def input_buffer_names(self) -> Sequence[str]:
        """Names of all input buffers."""

    @abc.abstractmethod
    def init(self, inputs: Mapping[str, Buffer],
             sample_rate: Float[Array, ""]) -> PluginState:
        """Initializes the plugin, given the first batch of samples. Called once."""

    @abc.abstractmethod
    def update(
        self,
        state: PluginState,
        inputs: Mapping[str, Buffer],
        sample_rate: Float[Array, ""],
    ) -> tuple[PluginState, Mapping[str, Buffer]]:
        """Processes a frame of audio data.
        
        Args:
          state: State of the plugin.
          inputs: Input buffers.
          sample_rate: Sample rate.
  
        Returns:
          New state and output buffers.
        """


class EmptyState(eqx.Module):
    """Empty plugin state."""


class StatelessPlugin(Plugin[EmptyState]):
    """Plugin with no state."""

    def init(self, inputs: Mapping[str, Buffer],
             sample_rate: Float[Array, ""]) -> EmptyState:
        del inputs, sample_rate  # Unused.
        return EmptyState()

    def update(
        self,
        state: EmptyState,
        inputs: Mapping[str, Buffer],
        sample_rate: Float[Array, ""],
    ) -> tuple[EmptyState, Mapping[str, Buffer]]:
        del state  # Unused.
        return EmptyState(), self.process(inputs, sample_rate)

    @abc.abstractmethod
    def process(self, inputs: Mapping[str, Buffer],
                sample_rate: Float[Array, ""]) -> Mapping[str, Buffer]:
        """Processes a frame of audio data."""
