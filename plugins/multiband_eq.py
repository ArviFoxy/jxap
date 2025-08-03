"""Multiband equalizer."""

from absl import app
from absl import flags
import jax.numpy as jnp
from jaxtyping import Float, Array
import jxap


class MultibandEq(jxap.Plugin):
    """Multiband equalizer using peak filters."""

    gain_db: Float[Array, "filters"]
    bandpass: jxap.filters.Peak

    input_port = jxap.InputPort("input")
    output_port = jxap.OutputPort("output")

    def __init__(
        self,
        gain_db: Float[Array, "filters"],
        freq: Float[Array, "filters"],
        q: Float[Array, "filters"],
    ):
        self.gain_db = gain_db
        self.bandpass = jxap.filters.Peak(freq, q)

    def init(self, sample_rate: jxap.Constant):
        self.bandpass.init(sample_rate)

    def __call__(
        self,
        inputs: dict[jxap.InputPort, jxap.Sample],
        sample_rate: jxap.Constant,
    ) -> dict[jxap.OutputPort, jxap.Sample]:
        x = inputs[self.input_port]
        h = self.bandpass(x, sample_rate)
        gain = jxap.db_to_ampl(self.gain_db)
        y = x + jnp.dot(gain, h)
        return {self.output_port: y}


# --- Exporting Logic ---
_OUTPUT_PATH = flags.DEFINE_string("output_path", "plugins/multiband_eq.jxap",
                                   "Where to write the plugin.")


def main(_):
    n = 50
    freq = jnp.linspace(20, 5000, n)
    plugin = MultibandEq(
        gain_db=jnp.interp(
            freq,
            jnp.array([20, 100, 500, 2000, 5000]),
            jnp.array([9.0, 6.0, 3.0, 0.0, 0.0]),
        ),
        freq=freq,
        q=0.15 * jnp.ones_like(freq),
    )
    jxap.export.export_plugin(plugin).save(_OUTPUT_PATH.value)
    print(f"Saved plugin to {_OUTPUT_PATH.value}")


if __name__ == "__main__":
    app.run(main)
