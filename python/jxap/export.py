"""Exports audio plugins as MLIR using `jax.export`."""

import dataclasses
import os
from typing import Any, Sequence
import zipfile

from absl import logging
import jax
import jax.core as jcore
import jax.tree_util as jtu
import jax.export
from jaxtyping import PyTree
from flax import nnx
import numpy as np

from jxap import types

_ALL_PLATFORMS = ["cpu", "cuda", "rocm"]

DimSize = Any


@dataclasses.dataclass(slots=True)
class _Scope:
    jax_scope: jax.export.SymbolicScope
    dtype: np.dtype
    buffer_size: DimSize


def _make_scope(dtype: np.dtype = np.float32):
    buffer_size = "BufferSize"
    scope = jax.export.SymbolicScope([f"{buffer_size} >= 1"])
    return _Scope(jax_scope=scope,
                  dtype=dtype,
                  buffer_size=jax.export.symbolic_shape(buffer_size,
                                                        scope=scope)[0])


def _get_init_args_shape(plugin: types.Plugin, scope: _Scope):
    buffer_shapes = {}
    for input_name in plugin.input_ports:
        buffer_shapes[input_name] = jax.ShapeDtypeStruct((scope.buffer_size,),
                                                         scope.dtype)
    sample_rate_shape = jax.ShapeDtypeStruct((), scope.dtype)
    return buffer_shapes, sample_rate_shape


def _get_update_args_shape(
    plugin: types.Plugin,
    state_shape: PyTree[jax.ShapeDtypeStruct],
    scope: _Scope,
) -> PyTree[jax.ShapeDtypeStruct]:
    buffer_shapes = {}
    for input_name in plugin.input_ports:
        buffer_shapes[input_name] = jax.ShapeDtypeStruct((scope.buffer_size,),
                                                         scope.dtype)
    sample_rate_shape = jax.ShapeDtypeStruct((), scope.dtype)
    return state_shape, buffer_shapes, sample_rate_shape


class _Closure:
    value: Any | None = None


@dataclasses.dataclass(slots=True, frozen=True)
class PackagedPlugin:
    """Plugin packaged for export.

    In this format JAX functions have been exported to MLIR.
    """
    name: str
    init_mlir: str
    update_mlir: str
    input_buffer_names: Sequence[str]
    output_buffer_names: Sequence[str]

    def save(self, file_path: str) -> None:
        """Saves the plugin to a JXAP plugin file (zip)."""
        # `makedirs(dirname)` can fail for relative paths.
        file_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("name.txt", self.name)
            zip_file.writestr("init.mlir", self.init_mlir)
            zip_file.writestr("update.mlir", self.update_mlir)
            zip_file.writestr("input_buffer_names.txt",
                              "\n".join(self.input_buffer_names))
            zip_file.writestr("output_buffer_names.txt",
                              "\n".join(self.output_buffer_names))
        logging.info('Saved plugin to %s', file_path)


def _check_plugin_output_update_shape(pytree: PyTree, expected_dtype: np.dtype,
                                      buffer_size: DimSize) -> None:
    if not isinstance(pytree, tuple):
        raise ValueError(
            "Plugin update() must return a (state, outputs) tuple, was: {pytree}"
        )
    outputs_tree = pytree[1]
    if not isinstance(outputs_tree, dict):
        raise ValueError(
            f"Plugin update() outputs must be a dictionary, was: {outputs_tree}"
        )
    for key, child in outputs_tree.items():
        if not isinstance(child, jcore.ShapedArray):
            raise ValueError(
                f"Plugin update() output {key} must be a Jax array, was: {child}"
            )
        if child.dtype != expected_dtype:
            raise ValueError(
                f"Plugin update() output {key} must have dtype {expected_dtype}, was: {child}"
            )
        if child.shape != (buffer_size,):
            raise ValueError(
                f"Plugin update() output {key} must have shape ({buffer_size},), was: {child}"
            )


def export_plugin(plugin: types.Plugin,
                  dtype: np.dtype = np.float32,
                  platforms: Sequence[str] | None = None) -> PackagedPlugin:
    """Exports an audio plugin to a file.
    
    Args:
      plugin: The audio plugin to save.
      path: The path.
      dtype: DType for audio buffers.
      platforms: Target platforms. By default uses ["cpu", "cuda", "rocm"].
    """
    if platforms is None:
        platforms = _ALL_PLATFORMS

    name = repr(plugin)
    logging.info("Exporting plugin %s:", name)
    scope = _make_scope(dtype)

    initialized_graphdef = _Closure()
    graphdef, empty_state, static = nnx.split(plugin, types.State, ...)
    print("  graphdef:", graphdef)
    print("  empty_state:", empty_state)
    print("  static:", static)

    # TODO: remove `buffers`
    def _init_fn(buffers, sample_rate):
        del buffers  # Unused.
        runtime_plugin = nnx.merge(graphdef, empty_state, static)
        runtime_plugin.init(sample_rate)
        init_graphdef, init_state, _ = nnx.split(runtime_plugin, types.State,
                                                 ...)
        initialized_graphdef.value = init_graphdef
        return init_state

    init_args_shape = _get_init_args_shape(plugin, scope=scope)
    exported_init_fn = jax.export.export(jax.jit(_init_fn),
                                         platforms=platforms)(*init_args_shape)
    logging.info("  init input tree: %s", exported_init_fn.in_tree)
    logging.info("  init inputs: %s", exported_init_fn.in_avals)
    logging.info("  init output tree: %s", exported_init_fn.out_tree)
    logging.info("  init outputs: %s", exported_init_fn.out_avals)

    def _update_fn(state, buffers, sample_rate):
        state_pytree = jax.tree.unflatten(exported_init_fn.out_tree, state)

        def _update_step(carry_state, buffers_n):
            runtime_plugin = nnx.merge(initialized_graphdef.value, carry_state,
                                       static)
            y_n = runtime_plugin(buffers_n, sample_rate)
            _, new_carry_state, _ = nnx.split(runtime_plugin, types.State, ...)
            return new_carry_state, y_n

        new_state, outputs = jax.lax.scan(_update_step, state_pytree, buffers)
        return new_state, outputs

    state_shape = exported_init_fn.out_avals
    update_args_shape = _get_update_args_shape(plugin, state_shape, scope)
    exported_update_fn = jax.export.export(
        jax.jit(_update_fn), platforms=platforms)(*update_args_shape)
    logging.info("  update input tree: %s", exported_update_fn.in_tree)
    logging.info("  update inputs: %s", exported_update_fn.in_avals)
    logging.info("  update output tree: %s", exported_update_fn.out_tree)
    logging.info("  update outputs: %s", exported_update_fn.out_avals)

    # TODO: check all the input/output shapes.
    update_out_tree = jtu.tree_unflatten(exported_update_fn.out_tree,
                                         exported_update_fn.out_avals)
    _check_plugin_output_update_shape(update_out_tree, dtype, scope.buffer_size)

    output_buffer_names = list(update_out_tree[1].keys())
    logging.info('  input buffer names: %s', plugin.input_ports)
    logging.info('  output buffer names: %s', output_buffer_names)

    output_ports = plugin.output_ports
    for output_buffer_name in output_buffer_names:
        if output_buffer_name not in output_ports:
            raise ValueError(
                f"Plugin outputs a buffer for a port defined outside the plugin: {output_buffer_name}"
            )

    return PackagedPlugin(
        name=name,
        init_mlir=exported_init_fn.mlir_module(),
        update_mlir=exported_update_fn.mlir_module(),
        input_buffer_names=plugin.input_ports,
        output_buffer_names=output_buffer_names,
    )
