"""Utilities for decibel scale."""

import jax
import jax.numpy as jnp


def db_to_ampl(x: jax.Array):
    """Convert decibels to amplitude."""
    return jnp.power(10.0, x / 20.0)


def ampl_to_db(x: jax.Array):
    """Convert amplitude to decibels.""" ""
    return 20.0 * jnp.log10(x)
