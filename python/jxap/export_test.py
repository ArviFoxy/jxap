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

    input: types.InputPort = types.InputPort("input")
    output: types.OutputPort = types.OutputPort("output")

    last_sample: types.State[jax.Array]
    last_buffer: types.State[jax.Array]

    def init(self, inputs: dict[types.InputPort, types.Buffer], sample_rate):
        del sample_rate  # Unused.
        self.last_sample = types.State(0.0)
        self.last_buffer = types.State(jnp.zeros_like(inputs[self.input]))

    def process(self, inputs: dict[types.InputPort, types.Buffer],
                sample_rate) -> dict[types.OutputPort, types.Buffer]:
        del sample_rate  # Unused.
        x = inputs[self.input]
        x_1 = jnp.concatenate([self.last_sample.value[jnp.newaxis], x[:-1]],
                              axis=0)
        y = x + x_1 + self.last_buffer.value
        self.last_sample.value = x[-1]
        self.last_buffer.value = x
        return {self.output: y}


class ExportingTest(testing.TestCase):

    def test_export_plugin(self):
        plugin = TestPlugin()
        print(plugin)
        print(jax.tree.structure(plugin))
        print(plugin.input_ports)
        print(plugin.output_ports)
        print(jax.tree.flatten(plugin))
        path = os.path.join(absltest.TEST_TMPDIR.value, "test_plugin.jxap")
        export.export_plugin(plugin).save(path)


if __name__ == '__main__':
    absltest.main()
