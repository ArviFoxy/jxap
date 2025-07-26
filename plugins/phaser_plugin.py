"""Phaser filter."""

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import jxap


class PhaserPlugin(jxap.Plugin):
    """A simple all-pass filter to create a phase shift effect."""

    # The center frequency of the filter in Hz. This controls the filter's response.
    center_freq_hz: float = 440.0

    # Names of the input and output buffers. These will correspond to Pipewire
    # port names. All ports are single channel but there can be any number of ports.
    input = jxap.InputPort("input")
    output = jxap.OutputPort("output")

    # State of the audio filter. Variables that change are wrapped with "jxap.State".
    last_input: jxap.State[jax.Array]
    last_output: jxap.State[jax.Array]

    def init(self, inputs, sample_rate):
        """Initializes the filter's state with silence."""
        del inputs, sample_rate  # Unused.
        self.last_input = jxap.State(0.0)
        self.last_output = jxap.State(0.0)

    def process(
        self,
        inputs: dict[jxap.InputPort, jxap.Buffer],
        sample_rate: Float[Array, ""],
    ) -> dict[jxap.OutputPort, jxap.Buffer]:
        """Processes one buffer of audio. All samples are float32."""
        # Calculate the filter coefficient 'alpha' from the desired center frequency.
        # This makes the filter's effect consistent across different sample rates.
        # All of this computation will be inlined by the JIT compiler.
        tan_theta = jnp.tan(jnp.pi * self.center_freq_hz / sample_rate)
        alpha = (1.0 - tan_theta) / (1.0 + tan_theta)

        def allpass_step(carry, x_n):
            """Processes a single sample through the all-pass filter."""
            x_prev, y_prev = carry
            y_n = alpha * x_n + x_prev - alpha * y_prev
            return (x_n, y_n), y_n

        # `jxap.Buffer` is just an alias for a Jax array with one dimension.
        input_buffer: Float[Array, "BufferSize"] = inputs[self.input]

        # Use jax.lax.scan for efficient vectorized processing of the filter.
        initial_state = (self.last_input.value, self.last_output.value)
        (final_input, final_output), output_buffer = jax.lax.scan(
            allpass_step,
            initial_state,
            input_buffer,
        )
        # Update the plugin's state.
        self.last_input.value = final_input
        self.last_output.value = final_output
        return {self.output: output_buffer}


# --- Exporting Logic ---
_OUTPUT_PATH = flags.DEFINE_string("output_path", "plugins/phaser_plugin.jxap",
                                   "Where to write the plugin.")


def main(_):
    plugin = PhaserPlugin()
    jxap.export.export_plugin(plugin).save(_OUTPUT_PATH.value)
    print(f"Saved plugin to {_OUTPUT_PATH.value}")


if __name__ == "__main__":
    app.run(main)
