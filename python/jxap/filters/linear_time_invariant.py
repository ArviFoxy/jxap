"""Linear time-invariant filters.

All implementations support batching over filter parameters.
"""

import jax
from jax import numpy as jnp
from jaxtyping import Float, Array

from jxap import types


def _validate_fs(fs, allow_none=True):
    """
    Check if the given sampling frequency is a scalar and raises an exception
    otherwise. If allow_none is False, also raises an exception for none
    sampling rates. Returns the sampling frequency as float or none if the
    input is none.
    """
    if fs is None:
        if not allow_none:
            raise ValueError("Sampling frequency can not be none.")
    else:  # should be float
        if not jnp.isscalar(fs):
            raise ValueError("Sampling frequency fs must be a single scalar.")
    return fs


def _iirnotch(w0, Q, fs=2.0):
    return _design_notch_peak_filter(w0, Q, "notch", fs)


def _iirpeak(w0, Q, fs):
    return _design_notch_peak_filter(w0, Q, "peak", fs)


def _design_notch_peak_filter(w0, Q, ftype, fs=2.0):
    """
    Design notch or peak digital filter.

    Parameters
    ----------
    w0 : float
        Normalized frequency to remove from a signal. If `fs` is specified,
        this is in the same units as `fs`. By default, it is a normalized
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : str
        The type of IIR filter to design:

            - notch filter : ``notch``
            - peak filter  : ``peak``
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0:

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.
    """
    fs = _validate_fs(fs, allow_none=False)

    w0 = 2 * w0 / fs

    # Checks if w0 is within the range
    #if w0 > 1.0 or w0 < 0.0:
    #    raise ValueError("w0 should be such that 0 < w0 < 1")

    # Get bandwidth
    bw = w0 / Q

    # Normalize inputs
    bw = bw * jnp.pi
    w0 = w0 * jnp.pi

    if ftype not in ("notch", "peak"):
        raise ValueError("Unknown ftype.")

    # Compute beta according to Eqs. 11.3.4 (p.575) and 11.3.19 (p.579) from
    # reference [1]. Due to assuming a -3 dB attenuation value, i.e, assuming
    # gb = 1 / jnp.sqrt(2), the following terms simplify to:
    #   (jnp.sqrt(1.0 - gb**2.0) / gb) = 1
    #   (gb / jnp.sqrt(1.0 - gb**2.0)) = 1
    beta = jnp.tan(bw / 2.0)

    # Compute gain: formula 11.3.6 (p.575) from reference [1]
    gain = 1.0 / (1.0 + beta)

    # Compute numerator b and denominator a
    # formulas 11.3.7 (p.575) and 11.3.21 (p.579)
    # from reference [1]
    if ftype == "notch":
        b = gain * jnp.array([1.0, -2.0 * jnp.cos(w0), 1.0])
    else:
        b = (1.0 - gain) * jnp.array([1.0, 0.0, -1.0])
    a = jnp.array([1.0, -2.0 * gain * jnp.cos(w0), (2.0 * gain - 1.0)])

    return b, a


class FIR(types.Filter[types.Sample, Float[Array, "..."]]):
    """Finite Impulse Response (FIR) filter.
    
    Warning: This is not sample-rate independent.
    """

    # Coefficients for the FIR filter.
    a: Float[Array, "... order+1"]

    # State for the FIR filter.
    prev_xs: types.State[Float[Array, "order"]]

    def __init__(self, a: Float[Array, "... order+1"]):
        self.a = a

    def init(self, sample_rate):
        del sample_rate  # Unused.
        order = self.a.shape[-1] - 1
        self.prev_xs = types.State(jnp.zeros((order,)))

    def __call__(
        self,
        x: types.Sample,
        sample_rate: types.Constant,
    ) -> types.Sample:
        """Processes an input sample through the FIR filter."""
        del sample_rate  # Unused.

        def _apply(a, xs):
            return jnp.dot(a, xs)

        # Vectorize over batch dimensions.
        apply_fn = _apply
        for _ in range(len(self.a.shape) - 1):
            apply_fn = jax.vmap(apply_fn, in_axes=(0, None), out_axes=0)

        # Shift previous inputs and add the new input.
        xs = jnp.concatenate(([x], self.prev_xs.value))
        self.prev_xs.value = xs[:-1]

        # Apply the FIR filter.
        y = apply_fn(self.a, xs)
        return y


class IIR(types.Filter[types.Sample, Float[Array, "..."]]):
    """Infinite Impulse Response (IIR) filter.
    
    Warning: This is not sample-rate independent.
    """

    # Coefficients for the IIR filter.
    a: Float[Array, "... forward_order+1"]
    b: Float[Array, "... back_order"]

    # State for the IIR filter.
    prev_xs: types.State[Float[Array, "forward_order"]]
    prev_ys: types.State[Float[Array, "... back_order"]]

    def __init__(
        self,
        a: Float[Array, "... forward_order+1"],
        b: Float[Array, "... back_order"],
    ):
        self.a = a
        self.b = b

    def init(self, sample_rate):
        del sample_rate  # Unused.
        forward_order = self.a.shape[-1] - 1
        back_order = self.b.shape[-1]
        batch_shape = self.b.shape[:-1]
        self.prev_xs = types.State(jnp.zeros((forward_order,)))
        self.prev_ys = types.State(
            jnp.zeros(tuple(list(batch_shape) + [back_order])))

    def __call__(
        self,
        x: types.Sample,
        sample_rate: types.Constant,
    ) -> types.Sample:
        """Processes an input sample through the IIR filter."""
        del sample_rate  # Unused.

        def _apply(a, b, xs, ys):
            return jnp.dot(a, xs) + jnp.dot(b, ys)

        # Vectorize over batch dimensions.
        apply_fn = _apply
        for _ in range(len(self.a.shape) - 1):
            apply_fn = jax.vmap(apply_fn, in_axes=(0, 0, None, 0), out_axes=0)

        # Shift previous inputs and add the new input.
        xs = jnp.concatenate((jnp.array([x]), self.prev_xs.value))
        self.prev_xs.value = xs[:-1]

        # Apply the IIR filter.
        ys = self.prev_ys.value
        y = apply_fn(self.a, self.b, xs, ys)

        # Update the outputs.
        ys = jnp.concat(
            [
                jnp.expand_dims(y, axis=-1),
                jax.lax.slice_in_dim(ys, 0, -1, axis=-1)
            ],
            axis=-1,
        )
        self.prev_ys.value = ys

        return y


class Peak(types.Filter[types.Sample, Float[Array, "..."]]):
    """Peak band-pass filter.

    Second-order IIR filter that 
    """

    # Center frequency.
    freq: Float[Array, "..."]
    # Quality factor - bandwidth relative to frequency.
    q: Float[Array, "..."]

    # State for the IIR filter.
    iir: types.State[IIR]

    def __init__(self, freq: Float[Array, "..."], q: Float[Array, "..."]):
        self.freq = freq
        self.q = q

    def init(self, sample_rate):
        peak_fn = _iirpeak
        for _ in range(len(self.freq.shape)):
            peak_fn = jax.vmap(peak_fn, in_axes=(0, 0, None), out_axes=0)
        b, a = peak_fn(self.freq, self.q, sample_rate)
        self.iir = types.State(IIR(a=a, b=b))
        self.iir.value.init(sample_rate)

    def __call__(self, inputs, sample_rate):
        return self.iir.value(inputs, sample_rate)
