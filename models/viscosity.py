"""Temperature-dependent water viscosity.

Uses the Vogel equation (a standard fluid-mechanics correlation for the
dynamic viscosity of water):

    mu(T) [Pa*s] = 2.414e-5 * 10 ** (247.8 / (T + 133.15))    (T in degC)

This reproduces measured water viscosity closely across the normal
membrane-operating range (e.g. mu(20 degC) = 1.002 mPa*s, mu(25 degC) =
0.891 mPa*s, both within ~0.2% of published reference values). It is
used to temperature-correct flux calculations, mirroring the standard
membrane-testing practice of normalizing flux to a reference
temperature (commonly 20 or 25 degC) via the ratio of viscosities,
since flux is inversely proportional to viscosity for a fixed pressure
and hydraulic resistance.
"""

import math


def water_viscosity_Pa_s(temperature_C):
    """Dynamic viscosity of pure water at `temperature_C`, in Pa*s."""
    if temperature_C < 0 or temperature_C > 100:
        raise ValueError("Temperature must be between 0 and 100 degC for this correlation.")
    return 2.414e-5 * 10 ** (247.8 / (temperature_C + 133.15))


def temperature_correction_factor(temperature_C, reference_temperature_C=25.0):
    """Factor = mu(reference)/mu(T). Multiply a permeability/flux value
    that was measured or calibrated at `reference_temperature_C` by
    this factor to predict what it would be at the actual operating
    temperature `temperature_C`.

    Water is less viscous when warm, so for a fixed pressure and
    resistance, flux is *higher* than the reference-temperature value
    when operating warmer than the reference (factor > 1), and *lower*
    when operating colder (factor < 1).
    """
    return water_viscosity_Pa_s(reference_temperature_C) / water_viscosity_Pa_s(temperature_C)
