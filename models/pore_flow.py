"""Detailed (structure-based) flux model using the Hagen-Poiseuille
equation for laminar flow through cylindrical pores — a standard way to
relate microfiltration/ultrafiltration flux to membrane structural
parameters (thickness, porosity, pore size), rather than lumping all
of that into a single empirical permeability coefficient the way
models.flux.permeability_flux does.

Pure-water permeance:

    Lp [m/(s*Pa)] = (eps * r^2) / (8 * mu * tau * delta)

where eps = porosity (0-1), r = pore radius (m), mu = viscosity
(Pa*s), tau = pore tortuosity (dimensionless, >=1; 1 = straight
pores), delta = membrane thickness (m). Flux is then

    J [m/s] = Lp * dP

This is the standard capillary-pore model used for symmetric
porous membranes (e.g. Cheryan's "Ultrafiltration and Microfiltration
Handbook", or any transport-phenomena membrane textbook); it is only
strictly valid for that idealized pore geometry, so treat results as
illustrative, like the rest of this MVP. Tortuosity is not one of the
user-supplied membrane fields, so a fixed literature-typical value
(DEFAULT_TORTUOSITY = 2.5, a commonly cited mid-range value for
porous membranes) is used unless overridden.
"""

DEFAULT_TORTUOSITY = 2.5


def pore_flow_permeance_LMH_bar(porosity, pore_size_nm, thickness_um, viscosity_Pa_s,
                                tortuosity=DEFAULT_TORTUOSITY):
    """Pure-water permeance implied by membrane structure, in LMH/bar
    (the same units as `Membrane.permeability_LMH_bar`, so the result
    can be compared directly against — or substituted for — the
    user-supplied empirical permeability value).
    """
    if not 0 < porosity <= 1:
        raise ValueError("Porosity must be between 0 (exclusive) and 1.")
    if pore_size_nm <= 0 or thickness_um <= 0 or viscosity_Pa_s <= 0:
        raise ValueError("Pore size, thickness and viscosity must be positive.")
    if tortuosity < 1:
        raise ValueError("Tortuosity must be >= 1.")

    r = (pore_size_nm / 2) * 1e-9   # nm -> m, diameter -> radius assumed given as pore "size" (diameter)
    delta = thickness_um * 1e-6      # um -> m

    Lp_SI = (porosity * r ** 2) / (8 * viscosity_Pa_s * tortuosity * delta)   # m/(s*Pa)

    # Convert m/(s*Pa) -> LMH/bar:
    #   1 m/(s*Pa) = 3.6e6 LMH per Pa = 3.6e6 * 1e5 LMH/bar = 3.6e11 LMH/bar
    return Lp_SI * 3.6e11


def pore_flow_flux_LMH(porosity, pore_size_nm, thickness_um, viscosity_Pa_s, TMP_bar,
                       tortuosity=DEFAULT_TORTUOSITY):
    """Flux [LMH] predicted directly from membrane structure and
    pressure, bypassing the user-supplied empirical permeability
    coefficient entirely."""
    if TMP_bar < 0:
        raise ValueError("Pressure must be non-negative.")
    Lp = pore_flow_permeance_LMH_bar(porosity, pore_size_nm, thickness_um, viscosity_Pa_s, tortuosity)
    return Lp * TMP_bar
