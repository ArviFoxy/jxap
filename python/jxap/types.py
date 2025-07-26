"""Common types for audio plugins."""

from __future__ import annotations

import abc
from typing import Any, Mapping, TypeVar

import jax
from jaxtyping import Array, Float
from flax import nnx

# Audio buffer type.
# Samples are ordered so that:
#   - buffer[0] is the earliest played sample.
#   - buffer[1] is the second earliest played sample.
#   - buffer[-1] is the latest.
Buffer = Float[Array, "BufferSize"]

# Re-exported for convenience.
Module = nnx.Module


@jax.tree_util.register_static
class InputPort(str):
    """Defines an input port (stream) of a plugin."""


@jax.tree_util.register_static
class OutputPort(str):
    """Defines an output port (stream) of a plugin."""


_PortType = TypeVar("PortType")


def _find_ports(obj: Any, cls: type[_PortType]) -> list[_PortType]:
    ports = set()
    # Look at class attributes
    for value in vars(type(obj)).values():
        if isinstance(value, cls):
            ports.add(value)
    # Look at instance attributes
    for value in vars(obj).values():
        if isinstance(value, cls):
            ports.add(value)
    return sorted(ports)


class State(nnx.Variable[nnx.A]):
    """Audio plugin state."""


class Plugin(nnx.Module):
    """Audio plugin. Plugins can be stateful using `State` and the `nnx` library."""

    @property
    def input_ports(self) -> list[InputPort]:
        return _find_ports(self, InputPort)

    @property
    def output_ports(self) -> list[OutputPort]:
        return _find_ports(self, OutputPort)

    def init(self, inputs: Mapping[InputPort, Buffer], sample_rate: Float[Array,
                                                                          ""]):
        """Initializes the plugin, given the first batch of samples. Called once."""
        del inputs, sample_rate  # Unused.
        # Does nothing by default.

    @abc.abstractmethod
    def process(
        self,
        inputs: Mapping[InputPort, Buffer],
        sample_rate: Float[Array, ""],
    ) -> Mapping[OutputPort, Buffer]:
        """Processes a frame of audio data.

        Args:
          state: State of the plugin.
          inputs: Input buffers.
          sample_rate: Sample rate [Hz], i.e. 1/44100 Hz.

        Returns:
          New state and output buffers.
        """
