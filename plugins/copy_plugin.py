"""Plugin that simply copies the input stream."""

from absl import app
from absl import flags

import jxap

_OUTPUT_PATH = flags.DEFINE_string("output_path", "plugins/copy_plugin.jxap",
                                   "Where to write the plugin.")


class CopyPlugin(jxap.Plugin):
    """Copies an audio stream."""

    input = jxap.InputPort("input")
    output = jxap.OutputPort("output")

    def __call__(self, inputs, sample_rate):
        del sample_rate  # Unused.
        return {self.output: inputs[self.input]}


def main(_):
    plugin = CopyPlugin()
    path = _OUTPUT_PATH.value
    jxap.export.export_plugin(plugin).save(path)
    print(f"Saved plugin to {path}")


if __name__ == "__main__":
    app.run(main)
