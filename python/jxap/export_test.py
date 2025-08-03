"""Tests for plugin exporting."""

import os

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp

from jxap import types
from jxap import export
from jxap import testing

FLAGS = flags.FLAGS


class TestPlugin(types.Plugin):

    input = types.InputPort("input")
    output = types.OutputPort("output")

    last_sample: types.State[jax.Array]

    def init(self, sample_rate: types.Constant):
        del sample_rate  # Unused.
        self.last_sample = types.State(jnp.array(0.0))

    def __call__(
        self,
        inputs: dict[types.InputPort, types.Sample],
        sample_rate: types.Constant,
    ) -> dict[types.OutputPort, types.Sample]:
        del sample_rate  # Unused.
        x = inputs[self.input]
        y = x + self.last_sample.value
        self.last_sample.value = x
        return {self.output: y}


class ExportingTest(testing.TestCase):

    def test_export_plugin(self):
        plugin = TestPlugin()
        print(plugin)
        path = os.path.join(absltest.TEST_TMPDIR.value, "test_plugin.jxap")
        plugin_package = export.export_plugin(plugin)
        print(plugin_package.update_mlir)
        plugin_package.save(path)


if __name__ == '__main__':
    absltest.main()
