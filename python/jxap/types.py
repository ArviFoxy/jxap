"""Common types for audio plugins."""

from __future__ import annotations

import abc
from typing import Any, Generic, Mapping, TypeVar

import jax
from jaxtyping import Array, Float
from flax import nnx

# Audio sample - a float32 scalar.
Sample = Float[Array, ""]

# Audio buffer type.
# Samples are ordered so that:
#   - buffer[0] is the earliest played sample.
#   - buffer[1] is the second earliest played sample.
#   - buffer[-1] is the latest.
Buffer = Float[Array, "BufferSize"]

# A constant value, e.g. a sample rate or buffer size.
# TODO: This will become a primitive type to handle dynamic shapes.
Constant = Float[Array, ""]


# State is a mutable variable that can be updated during processing.
# For example, it can be used to store the last processed sample or
# a delay buffer.
class State(nnx.Variable[nnx.A]):
    """Audio plugin state."""


class Module(nnx.Module):
    """Base class for all audio modules (filters and plugins).
    
    Modules can be stateful using `State` and the `nnx` library.
    """

    def init(self, sample_rate: Constant):
        """Initializes the module. Called once."""
        del sample_rate  # Unused.
        # Does nothing by default.


FilterInput = TypeVar("FilterInput")
FilterOutput = TypeVar("FilterOutput")


class Filter(Module, Generic[FilterInput, FilterOutput]):
    """An input -> output transformation."""

    @abc.abstractmethod
    def __call__(
        self,
        inputs: FilterInput,
        sample_rate: Constant,
    ) -> FilterOutput:
        """Processes an input into an output.

        Args:
          inputs: The module input.
          sample_rate: Sample rate [Hz], i.e. 1/44100 Hz.

        Returns:
          The module output.
        """


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


class Plugin(Filter[Mapping[InputPort, Sample], Mapping[OutputPort, Sample]]):
    """Base class for audio plugin.

    An audio plugin is a `Filter` that can have multiple input and output ports.
    These ports are connected to audio streams in the audio processing graph.

    The `__call__` method  receives a dict of samples as an input and returns a dict of samples as an output.

    As a `Filter`, it can have mutable state variables that are updated during processing.

    Plugins can be exported to a JXAP plugin package.
    """

    @property
    def input_ports(self) -> list[InputPort]:
        """Returns all input ports defined in the plugin."""
        return _find_ports(self, InputPort)

    @property
    def output_ports(self) -> list[OutputPort]:
        """Returns all output ports defined in the plugin."""
        return _find_ports(self, OutputPort)

    @abc.abstractmethod
    def __call__(
        self,
        inputs: dict[InputPort, Buffer],
        sample_rate: Constant,
    ) -> dict[OutputPort, Buffer]:
        """Processes audio samples.

        The plugin state is updated in-place.

        Args:
          inputs: The input audio samples, one for each input port.
          sample_rate: Sample rate [Hz], i.e. 1/44100 Hz.

        Returns:
          The output audio samples, one for each output port.
        """
