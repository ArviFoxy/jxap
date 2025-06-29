"""Plugin that simply copies the input stream."""

from absl import app
from absl import flags

import jxap

_OUTPUT_PATH = flags.DEFINE_string("output_path", "plugins/copy_plugin.jxap",
                                   "Where to write the plugin.")


class CopyPlugin(jxap.StatelessPlugin):
    """Copies an audio stream."""

    input_name: str = "input"
    output_name: str = "output"

    @property
    def input_buffer_names(self):
        return [self.input_name]

    def process(self, inputs: dict[str, jxap.Buffer],
                sample_rate: jxap.Float[jxap.Array, ""]):
        del sample_rate  # Unused.
        return {self.output_name: inputs[self.input_name]}


def main(_):
    plugin = CopyPlugin()
    path = _OUTPUT_PATH.value
    jxap.export.export_plugin(plugin).save(path)
    print(f"Saved plugin to {path}")


if __name__ == "__main__":
    app.run(main)
