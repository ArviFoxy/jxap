"""A stereo flanger audio effect with feedback and LFO modulation."""

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import jxap

# Define a maximum delay time for the flanger in seconds.
# This determines the size of our delay buffer.
MAX_DELAY_SECONDS = 0.02  # 20 milliseconds


class FlangerPlugin(jxap.Plugin):
    """A flanger effect using a modulated delay line."""

    # --- Controllable Parameters ---
    rate_hz: float = 0.5
    depth_s: float = 0.005
    feedback: float = 0.7
    mix: float = 0.5

    # --- Ports ---
    input = jxap.InputPort("input")
    output = jxap.OutputPort("output")

    # --- Internal State ---
    lfo_phase: jxap.State[jax.Array]
    delay_buffer: jxap.State[jax.Array]
    write_head: jxap.State[jax.Array]

    def init(self, inputs, sample_rate: Float[Array, ""]):
        """Initializes the plugin's state."""
        del inputs
        # TODO: Handle dynamic shapes somehow. This is not an issue on StableHLO side,
        # only a limitation of `jax.jit`.
        buffer_size = int(MAX_DELAY_SECONDS * 44100)

        self.lfo_phase = jxap.State(jnp.asarray(0.0))
        self.delay_buffer = jxap.State(jnp.zeros(buffer_size))
        self.write_head = jxap.State(jnp.asarray(0, dtype=jnp.int32))

    def process(
        self,
        inputs: dict[jxap.InputPort, jxap.Buffer],
        sample_rate: Float[Array, ""],
    ) -> dict[jxap.OutputPort, jxap.Buffer]:
        """Processes one buffer of audio."""

        def flanger_step(carry, x_n):
            """Processes a single sample."""
            lfo_phase, write_head, delay_buffer = carry
            buffer_size = len(delay_buffer)

            # 1. Calculate LFO value
            lfo_increment = 2.0 * jnp.pi * self.rate_hz / sample_rate
            new_lfo_phase = lfo_phase + lfo_increment
            lfo_val = jnp.sin(new_lfo_phase)

            # 2. Calculate modulated delay time in samples
            center_delay = self.depth_s * sample_rate
            modulated_delay = center_delay * (1.0 + lfo_val) / 2.0

            # 3. Read from the delay buffer using manual linear interpolation
            read_head_float = (write_head - modulated_delay) % buffer_size
            read_head_idx1 = jnp.floor(read_head_float).astype(jnp.int32)
            read_head_idx2 = (read_head_idx1 + 1) % buffer_size
            interpolation_frac = read_head_float - jnp.floor(read_head_float)

            delayed_sample = (delay_buffer[read_head_idx1] *
                              (1.0 - interpolation_frac) +
                              delay_buffer[read_head_idx2] * interpolation_frac)

            # 4. Write to the delay buffer
            feedback_sample = delayed_sample * self.feedback
            buffer_input = jnp.clip(x_n + feedback_sample, -1.0, 1.0)
            new_delay_buffer = delay_buffer.at[write_head].set(buffer_input)

            # 5. Calculate the final output sample
            y_n = (1.0 - self.mix) * x_n + self.mix * delayed_sample

            # 6. Update state
            new_write_head = (write_head + 1) % buffer_size
            new_carry = (new_lfo_phase, new_write_head, new_delay_buffer)

            return new_carry, y_n

        initial_state = (
            self.lfo_phase.value,
            self.write_head.value,
            self.delay_buffer.value,
        )
        input_buffer = inputs[self.input]

        (final_lfo_phase, final_write_head,
         final_delay_buffer), output_buffer = (jax.lax.scan(
             flanger_step,
             initial_state,
             input_buffer,
         ))

        self.lfo_phase.value = final_lfo_phase
        self.write_head.value = final_write_head
        self.delay_buffer.value = final_delay_buffer

        return {self.output: output_buffer}


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
