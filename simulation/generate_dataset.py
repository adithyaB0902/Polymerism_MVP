"""Generate a synthetic ("virtual experiment") dataset by sampling random
membrane/water/operating-condition combinations and running them through
`models.simulator.run_simulation`.

This is physics-model output, not experimental data (see
docs/assumptions.md: "Synthetic data is not experimental evidence").
It is used to power the Pareto/optimization/ML-training tabs in the app.

Sampling ranges for TMP_bar, permeability_LMH_bar and thickness_um are
loaded from config/parameter_ranges.yaml so that file — previously
defined but never actually read by any code — is the single source of
truth for those three ranges, instead of duplicating the same numbers
here and risking the two drifting apart. All other ranges are still
defined inline below; if you add a parameter to the YAML file, add a
matching lookup here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from models.membrane import Membrane, Water, OperatingConditions, Targets
from models.simulator import run_simulation

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "parameter_ranges.yaml"

# Fallback ranges, used only if config/parameter_ranges.yaml is missing
# or malformed, so dataset generation never hard-fails on a config issue.
_DEFAULT_RANGES = {
    "TMP_bar": {"minimum": 0.2, "maximum": 20},
    "permeability_LMH_bar": {"minimum": 1, "maximum": 100},
    "thickness_um": {"minimum": 10, "maximum": 500},
}


def _load_parameter_ranges():
    try:
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        ranges = config.get("parameters", {})
        # Fill in any parameter missing from the file with the fallback.
        return {**_DEFAULT_RANGES, **{k: v for k, v in ranges.items() if k in _DEFAULT_RANGES}}
    except (FileNotFoundError, yaml.YAMLError):
        return _DEFAULT_RANGES


def generate_virtual_experiments(n=500, seed=42):
    """Sample `n` random candidates and return their simulated results as
    a DataFrame, one row per candidate. Sampling is deterministic for a
    given `seed`."""
    ranges = _load_parameter_ranges()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        m = Membrane("Custom Polymer", rng.choice(["MF","UF","NF","RO"]),
                     rng.uniform(ranges["thickness_um"]["minimum"], ranges["thickness_um"]["maximum"]),
                     rng.uniform(ranges["permeability_LMH_bar"]["minimum"], ranges["permeability_LMH_bar"]["maximum"]),
                     rng.uniform(1,100), rng.uniform(500,100000),
                     rng.uniform(.1,.8), rng.uniform(-1,1), rng.uniform(0,1),
                     10000, rng.uniform(50,99.9), rng.uniform(.005,.15))
        w = Water("Custom contaminant", rng.uniform(10,1000), rng.uniform(0,100),
                  rng.uniform(100,3000), rng.uniform(4,10), rng.uniform(10,40))
        o = OperatingConditions(rng.uniform(ranges["TMP_bar"]["minimum"], ranges["TMP_bar"]["maximum"]), rng.uniform(1,50),
                                rng.uniform(.01,1), rng.uniform(5,80),
                                rng.uniform(.5,24), w.temperature_C, rng.uniform(.1,10), .7, 8)
        t = Targets(95,30,20,2,10)
        r = run_simulation(m,w,o,t)
        rows.append({**m.__dict__, **{f"water_{k}":v for k,v in w.__dict__.items()},
                     **{f"op_{k}":v for k,v in o.__dict__.items()}, **r,
                     "dataset_type":"synthetic_physics"})
    return pd.DataFrame(rows)
