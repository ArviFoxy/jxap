"""A flanger audio effect with feedback and LFO modulation."""

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import jxap


class FlangerPlugin(jxap.Plugin):
    """A flanger effect using a modulated delay line."""

    # --- Controllable Parameters ---
    max_delay_s: float = 0.02  # 20 milliseconds
    rate_hz: float = 0.5  # 0.5 Hz oscillation
    depth_s: float = 0.005  # 5 milliseconds depth
    feedback: float = 0.7  # Feedback for the delay line
    mix: float = 0.5  # Mix of the original signal and the delayed signal

    # --- Ports ---
    input = jxap.InputPort("input")
    output = jxap.OutputPort("output")

    # --- Internal State ---
    lfo_phase: jxap.State[jax.Array]
    delay: jxap.VariableDelay

    def init(self, sample_rate: jxap.Constant):
        """Initializes the plugin's state."""
        self.delay = jxap.VariableDelay(max_seconds=self.max_delay_s)
        self.delay.init(sample_rate)
        self.lfo_phase = jxap.State(jnp.asarray(0.0))

    def __call__(
        self,
        inputs: dict[jxap.InputPort, jxap.Sample],
        sample_rate: jxap.Constant,
    ) -> dict[jxap.OutputPort, jxap.Sample]:
        """Processes one sample of audio."""

        x = inputs[self.input]

        # 1. Calculate LFO value
        lfo_increment = 2.0 * jnp.pi * self.rate_hz / sample_rate
        new_lfo_phase = (self.lfo_phase.value + lfo_increment) % (2.0 * jnp.pi)
        self.lfo_phase.value = new_lfo_phase
        lfo_val = jnp.sin(new_lfo_phase)

        # 2. Calculate modulated delay time in seconds
        modulated_delay = self.depth_s * (1.0 + lfo_val) / 2.0

        # 3. Read from the delay buffer
        delayed_sample = self.delay.read(modulated_delay, sample_rate)

        # 4. Apply feedback and write to the delay buffer
        feedback_sample = jnp.clip(x + delayed_sample * self.feedback, -1.0,
                                   1.0)
        self.delay.write(feedback_sample)

        # 5. Calculate the final output sample
        y = (1.0 - self.mix) * x + self.mix * delayed_sample

        return {self.output: y}


# --- Exporting Logic ---
_OUTPUT_PATH = flags.DEFINE_string("output_path", "plugins/flanger_plugin.jxap",
                                   "Where to write the plugin.")


def main(_):
    plugin = FlangerPlugin()
    packaged_plugin = jxap.export.export_plugin(plugin)
    print(packaged_plugin.init_mlir)
    packaged_plugin.save(_OUTPUT_PATH.value)
    print(f"Saved plugin to {_OUTPUT_PATH.value}")


if __name__ == "__main__":
    app.run(main)
