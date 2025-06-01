"""Exports audio plugins as MLIR using `jax.export`."""

import dataclasses
import os
from typing import Any, Sequence

from absl import logging
import jax
import jax.export
from jaxtyping import PyTree
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
    for input_name in plugin.input_buffer_names:
        buffer_shapes[input_name] = jax.ShapeDtypeStruct((scope.buffer_size,),
                                                         scope.dtype)
    sample_rate_shape = jax.ShapeDtypeStruct((), scope.dtype)
    return buffer_shapes, sample_rate_shape


def _get_update_args_shape(
    plugin: types.Plugin,
    state_shape: PyTree[jax.ShapeDtypeStruct],
    scope: _Scope,
) -> tuple[jax.export.SymbolicScope, Any]:
    buffer_shapes = {}
    for input_name in plugin.input_buffer_names:
        buffer_shapes[input_name] = jax.ShapeDtypeStruct((scope.buffer_size,),
                                                         scope.dtype)
    sample_rate_shape = jax.ShapeDtypeStruct((), scope.dtype)
    return state_shape, buffer_shapes, sample_rate_shape


class _TreeDefClosure:
    tree_def: Any | None = None


def export_plugin(plugin: types.Plugin,
                  path: str,
                  dtype: np.dtype = np.float32,
                  platforms: Sequence[str] | None = None) -> None:
    """Exports an audio plugin to a file.
    
    Args:
      plugin: The audio plugin to save.
      path: The path.
      dtype: DType for audio buffers.
      platforms: Target platforms. By default uses ["cpu", "cuda", "rocm"].
    """
    if platforms is None:
        platforms = _ALL_PLATFORMS

    logging.info("Exporting plugin %s:", type(plugin))
    scope = _make_scope(dtype)

    state_tree_def = _TreeDefClosure()

    def _init_fn(buffers, sample_rate):
        state, tree_def = jax.tree.flatten(plugin.init(buffers, sample_rate))
        state_tree_def.tree_def = tree_def
        return state

    init_args_shape = _get_init_args_shape(plugin, scope=scope)
    exported_init_fn = jax.export.export(jax.jit(_init_fn),
                                         platforms=platforms)(*init_args_shape)
    logging.info("  init input tree: %s", exported_init_fn.in_tree)
    logging.info("  init inputs: %s", exported_init_fn.in_avals)
    logging.info("  init output tree: %s", exported_init_fn.out_tree)
    logging.info("  init outputs: %s", exported_init_fn.out_avals)
    init_path = f"{path}-init"
    logging.info("Saving to %s...", init_path)
    os.makedirs(os.path.dirname(init_path), exist_ok=True)
    with open(init_path, 'w', encoding='utf-8') as out_file:
        out_file.write(exported_init_fn.mlir_module())

    def _update_fn(state, buffers, sample_rate):
        state = jax.tree.unflatten(state_tree_def.tree_def, state)
        new_state, outputs = plugin.update(state, buffers, sample_rate)
        new_state, _ = jax.tree.flatten(new_state)
        return new_state, outputs

    state_shape = exported_init_fn.out_avals
    update_args_shape = _get_update_args_shape(plugin, state_shape, scope)
    exported_update_fn = jax.export.export(
        jax.jit(_update_fn), platforms=platforms)(*update_args_shape)
    logging.info("  update input tree: %s", exported_update_fn.in_tree)
    logging.info("  update inputs: %s", exported_update_fn.in_avals)
    logging.info("  update output tree: %s", exported_update_fn.out_tree)
    logging.info("  update outputs: %s", exported_update_fn.out_avals)
    update_path = f"{path}-update"
    logging.info("Saving to %s...", update_path)
    os.makedirs(os.path.dirname(update_path), exist_ok=True)
    with open(update_path, 'w', encoding='utf-8') as out_file:
        out_file.write(exported_update_fn.mlir_module())
