"""Membrane "samples" as a lab concept: convert between the physics
layer's `Membrane` dataclass, the `samples` table in the lab database,
and the illustrative presets shipped in data/membrane_properties.csv.

This is what finally wires data/membrane_properties.csv into the app —
previously-unused reference data (see the earlier README note about it)
now backs a real "load a preset membrane" feature.
"""

import csv
from pathlib import Path

from models.membrane import Membrane
from db import repository as repo

PRESETS_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "membrane_properties.csv"

# Fields on Membrane that a sample row can supply; anything else keeps
# the Membrane dataclass's own default.
_MEMBRANE_FIELDS_FROM_SAMPLE = ["polymer_name", "membrane_type", "thickness_um", "permeability_LMH_bar",
                               "pore_size_nm", "MWCO", "porosity", "surface_charge", "hydrophilicity",
                               "membrane_area_cm2", "baseline_rejection_percent", "fouling_coefficient"]


def membrane_to_sample_fields(membrane):
    """Membrane dataclass -> dict suitable for db.repository.create_sample."""
    return {f: getattr(membrane, f) for f in _MEMBRANE_FIELDS_FROM_SAMPLE}


def sample_to_membrane(sample_row):
    """DB sample row (dict) -> a Membrane object."""
    kwargs = {f: sample_row[f] for f in _MEMBRANE_FIELDS_FROM_SAMPLE if sample_row.get(f) is not None}
    return Membrane(**kwargs)


def create_sample_from_membrane(conn, membrane, batch_id=None, notes=None, origin="manual",
                                calibrated_from_sample_id=None):
    """Persist a Membrane as a new sample row; returns the new sample id."""
    return repo.create_sample(conn, membrane_to_sample_fields(membrane), batch_id=batch_id, notes=notes,
                              origin=origin, calibrated_from_sample_id=calibrated_from_sample_id)


def load_presets(csv_path=None):
    """Read the illustrative membrane presets shipped with the project.
    Returns a list of dicts with polymer_name, membrane_type,
    permeability_LMH_bar, thickness_um, baseline_rejection_percent,
    source_type, source, notes."""
    path = Path(csv_path) if csv_path else PRESETS_CSV_PATH
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        for numeric_field in ["permeability_LMH_bar", "thickness_um", "baseline_rejection_percent"]:
            row[numeric_field] = float(row[numeric_field])
    return rows


def create_sample_from_preset(conn, preset, batch_id=None):
    """`preset` is one row (dict) from `load_presets()`. Returns the new
    sample id. Only the fields the preset CSV actually carries are set;
    everything else (pore size, porosity, fouling coefficient, etc.)
    keeps Membrane's defaults — presets are explicitly illustrative
    starting points, not complete measured characterizations (see
    data/membrane_properties.csv's own source_type/notes columns)."""
    membrane = Membrane(
        polymer_name=preset["polymer_name"], membrane_type=preset["membrane_type"],
        thickness_um=preset["thickness_um"], permeability_LMH_bar=preset["permeability_LMH_bar"],
        baseline_rejection_percent=preset["baseline_rejection_percent"],
    )
    notes = f"Loaded from preset: {preset.get('source', 'unknown source')}. {preset.get('notes', '')}".strip()
    return create_sample_from_membrane(conn, membrane, batch_id=batch_id, notes=notes, origin="preset")
