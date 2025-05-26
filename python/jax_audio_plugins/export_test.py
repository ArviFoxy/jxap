"""Tests for plugin exporting."""

import os

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp

import equinox as eqx
from jax_audio_plugins import types
from jax_audio_plugins import export
from jax_audio_plugins import test_utils

FLAGS = flags.FLAGS


class TestPluginState(eqx.Module):
    last_sample: jax.Array
    last_buffer: jax.Array


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
        x_1 = jnp.concatenate([x[1:], state.last_sample[jnp.newaxis]], axis=0)
        y = x + x_1 + state.last_buffer
        new_state = TestPluginState(last_sample=x[0], last_buffer=x)
        return new_state, {"output": y}


class ExportingTest(test_utils.TestCase):

    def test_export_plugin(self):
        plugin_config = TestPlugin()
        path = os.path.join(absltest.TEST_TMPDIR.value, "plugin.jxap")
        export.export_plugin(plugin_config, path)


if __name__ == '__main__':
    absltest.main()
