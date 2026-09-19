"""Storage and comparison helpers for laboratory characterization measurements."""

CHARACTERIZATION_FIELDS = ["ftir", "sem", "edx", "bet", "contact_angle", "porosity", "swelling", "acid_stability"]


def validate_characterization(record):
    return {key: record.get(key) for key in CHARACTERIZATION_FIELDS}

