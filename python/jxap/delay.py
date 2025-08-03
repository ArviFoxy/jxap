"""Delay line implementation"""

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from jxap import types


class Delay(types.Module):
    """Static delay line. Delays the input signal by a fixed duration.
  
    Example usage:
    ```
    delay = Delay(seconds=0.1)
    delay.init(sample_rate)
    
    feedback = 0.1
    for sample in [0.1, 0.2, 0.3, ...]:
      delayed_sample = delay.read(sample_rate)
      delay.write(sample + delayed_sample * feedback)
    ```
    """

    seconds: float

    # Maximum sample rate to work around issues with dynamic shapes in JAX.
    max_sample_rate: float = 320_000.0  # 320 kHz

    # Cyclic buffer for the delay line state.
    delay_buffer: types.State[jax.Array]
    write_head: types.State[jax.Array]

    def __init__(self, seconds: float):
        self.seconds = seconds

    def init(self, sample_rate: types.Constant):
        """Initializes the delay line state and returns the initial arrays."""
        del sample_rate  # Unused.
        # TODO: Handle dynamic shapes somehow. This is not an issue on StableHLO side,
        # only a limitation of `jax.jit`.
        max_buffer_size = int(self.max_sample_rate * self.seconds)
        self.delay_buffer = types.State(jnp.zeros(max_buffer_size))
        self.write_head = types.State(jnp.asarray(0, dtype=jnp.int32))

    def read(self, sample_rate: types.Constant):
        """Reads the delayed sample."""
        delay_length = int(self.seconds * sample_rate)
        read_head = (self.write_head.value -
                     delay_length) % self.delay_buffer.value.shape[0]
        return self.delay_buffer.value[read_head]

    def write(self, x_n: types.Sample):
        """Writes a sample to the delay buffer."""
        buffer_size = self.delay_buffer.value.shape[0]
        new_delay_buffer = self.delay_buffer.value.at[
            self.write_head.value].set(x_n)
        new_write_head = (self.write_head.value + 1) % buffer_size
        self.delay_buffer.value = new_delay_buffer
        self.write_head.value = new_write_head


class VariableDelay(nnx.Module):
    """Variable delay line. Delays the input signal by a varying duration up to a maximum.

    Uses linear interpolation to read from the delay buffer.
  
    Example usage:
    ```
    delay = VariableDelay(max_seconds=0.5)
    delay.init(sample_rate)
    
    feedback = 0.1
    samples = jnp.array([0.1, 0.2, 0.3, ...])
    delay_s = jnp.array([0.2, 0.22, 0.25, ...])
    for sample, delay_s_n in zip(samples, delay_s):
      delayed_sample = delay.read(delay_s, sample_rate)
      delay.write(sample + delayed_sample * feedback)
    ```
    """

    max_seconds: float

    # Maximum sample rate to work around issues with dynamic shapes in JAX.
    max_sample_rate: float = 320_000.0  # 320 kHz

    # Cyclic buffer for the delay line state.
    delay_buffer: types.State[jax.Array]
    write_head: types.State[jax.Array]

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds

    def init(self, sample_rate: types.Constant):
        """Initializes the delay line state and returns the initial arrays."""
        del sample_rate  # Unused.
        # TODO: Handle dynamic shapes somehow. This is not an issue on StableHLO side,
        # only a limitation of `jax.jit`.
        max_buffer_size = int(self.max_sample_rate * self.max_seconds)
        self.delay_buffer = types.State(jnp.zeros(max_buffer_size))
        self.write_head = types.State(jnp.asarray(0, dtype=jnp.int32))

    def read(
        self,
        delay_seconds: types.Sample,
        sample_rate: types.Constant,
    ) -> types.Sample:
        """Reads the delayed sample."""
        delay_length = delay_seconds * sample_rate
        buffer_size = self.delay_buffer.value.shape[0]
        read_head_float = (self.write_head.value - delay_length) % buffer_size
        read_head_idx1 = jnp.floor(read_head_float).astype(jnp.int32)
        read_head_idx2 = (read_head_idx1 + 1) % buffer_size
        interpolation_frac = read_head_float - jnp.floor(read_head_float)
        return (self.delay_buffer.value[read_head_idx1] *
                (1.0 - interpolation_frac) +
                self.delay_buffer.value[read_head_idx2] * interpolation_frac)

    def write(self, sample: types.Sample):
        """Writes a sample to the delay buffer."""
        buffer_size = self.delay_buffer.value.shape[0]
        new_delay_buffer = self.delay_buffer.value.at[
            self.write_head.value].set(sample)
        new_write_head = (self.write_head.value + 1) % buffer_size
        self.delay_buffer.value = new_delay_buffer
        self.write_head.value = new_write_head
