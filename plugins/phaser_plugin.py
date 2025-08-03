"""Phaser filter."""

from absl import app
from absl import flags
import jax.numpy as jnp
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
    last_input: jxap.State[jxap.Sample]
    last_output: jxap.State[jxap.Sample]

    def init(self, sample_rate: jxap.Constant):
        """Initializes the filter's state with silence."""
        del sample_rate  # Unused.
        self.last_input = jxap.State(0.0)
        self.last_output = jxap.State(0.0)

    def __call__(
        self,
        inputs: dict[jxap.InputPort, jxap.Sample],
        sample_rate: jxap.Constant,
    ) -> dict[jxap.OutputPort, jxap.Sample]:
        """Processes one buffer of audio. All samples are float32."""
        # Calculate the filter coefficient 'alpha' from the desired center frequency.
        # This makes the filter's effect consistent across different sample rates.
        # All of this computation will be inlined by the JIT compiler.
        tan_theta = jnp.tan(jnp.pi * self.center_freq_hz / sample_rate)
        alpha = (1.0 - tan_theta) / (1.0 + tan_theta)

        x_n = inputs[self.input]
        x_prev = self.last_input.value
        y_prev = self.last_output.value
        y_n = alpha * x_n + x_prev - alpha * y_prev
        self.last_input.value = x_n
        self.last_output.value = y_n
        return {self.output: y_n}


# --- Exporting Logic ---
_OUTPUT_PATH = flags.DEFINE_string("output_path", "plugins/phaser_plugin.jxap",
                                   "Where to write the plugin.")


def main(_):
    plugin = PhaserPlugin()
    jxap.export.export_plugin(plugin).save(_OUTPUT_PATH.value)
    print(f"Saved plugin to {_OUTPUT_PATH.value}")


if __name__ == "__main__":
    app.run(main)
