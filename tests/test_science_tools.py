import sqlite3

import numpy as np
import pandas as pd
import pytest

from db.repository import backup_database, get_connection, restore_database
from research.rsm.box_behnken import FACTORS, generate_box_behnken_design
from research.science_tools import (
    multiresponse_desirability,
    porosity,
    regeneration_summary,
    swelling_degree,
)


def test_swelling_and_porosity_formulas():
    assert swelling_degree(1.2, 1.0) == pytest.approx(20.0)
    assert porosity(1.2, 1.0, 1.0, 10.0, 2.0) == pytest.approx(1.0)


def test_multiresponse_desirability_and_regeneration():
    values = pd.DataFrame({"removal": [80.0, 90.0], "swelling": [20.0, 40.0]})
    result = multiresponse_desirability(values, {
        "removal": ("maximize", 0, 100, 1),
        "swelling": ("minimize", 0, 100, 1),
    })
    assert result.loc[1, "overall_desirability"] < result.loc[0, "overall_desirability"]
    reuse = regeneration_summary([
        {"cycle": 1, "removal_percent": 90, "desorbed_mg": 8, "loaded_mg": 10},
        {"cycle": 2, "removal_percent": 72, "desorbed_mg": 7, "loaded_mg": 10},
    ])
    assert reuse.loc[1, "retained_removal_percent"] == pytest.approx(80)
    assert reuse.loc[0, "desorption_efficiency_percent"] == pytest.approx(80)


def test_optional_extra_factor_preserves_default_design():
    assert len(generate_box_behnken_design()) == 29
    factors = dict(FACTORS)
    factors["Cu_mg_l"] = (0.0, 1.0, 2.0)
    assert len(generate_box_behnken_design(factors, center_points=3)) == 43


def test_database_backup_and_restore(tmp_path):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    restored = tmp_path / "restored.db"
    connection = get_connection(source)
    connection.execute("INSERT INTO batches (created_at, name) VALUES ('now', 'test')")
    connection.commit()
    backup_database(connection, backup)
    restore_database(backup, restored)
    with sqlite3.connect(restored) as check:
        assert check.execute("SELECT name FROM batches").fetchone()[0] == "test"
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.db"
        bad.write_bytes(b"not sqlite")
        restore_database(bad, restored)