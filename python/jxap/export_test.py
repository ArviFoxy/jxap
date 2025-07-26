"""Tests for plugin exporting."""

import dataclasses
import os

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp

import equinox as eqx
from jxap import types
from jxap import export
from jxap import testing

FLAGS = flags.FLAGS


class TestPluginState(eqx.Module):
    last_sample: jax.Array
    last_buffer: jax.Array


@dataclasses.dataclass()
class TestPlugin(types.Plugin[TestPluginState]):

    @property
    def input_buffer_names(self):
        return ["input"]

    def init(self, inputs: dict[str, types.Buffer],
             sample_rate) -> TestPluginState:
        del sample_rate  # Unused.
        return TestPluginState(last_sample=jnp.array(0.0),
                               last_buffer=jnp.zeros_like(inputs["input"]))

    def update(self, state: TestPluginState, inputs: dict[str, types.Buffer],
               sample_rate):
        del sample_rate  # Unused.
        x = inputs["input"]
        x_1 = jnp.concatenate([state.last_sample[jnp.newaxis], x[:-1]], axis=0)
        y = x + x_1 + state.last_buffer
        new_state = TestPluginState(last_sample=x[-1], last_buffer=x)
        return new_state, {"output": y}


class ExportingTest(testing.TestCase):

    def test_export_plugin(self):
        plugin_config = TestPlugin()
        path = os.path.join(absltest.TEST_TMPDIR.value, "test_plugin.jxap")
        export.export_plugin(plugin_config).save(path)


if __name__ == '__main__':
    absltest.main()
