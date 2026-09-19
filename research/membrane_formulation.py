"""CS-CA-biochar formulation records."""

from dataclasses import asdict, dataclass, field


@dataclass
class MembraneFormulation:
    membrane_id: str
    chitosan_wt_percent: float
    cellulose_acetate_wt_percent: float
    biochar_wt_percent: float
    fabrication_info: str = ""
    biochar_properties: dict = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        values = (self.chitosan_wt_percent, self.cellulose_acetate_wt_percent, self.biochar_wt_percent)
        if any(float(v) < 0 for v in values):
            raise ValueError("Formulation fractions cannot be negative.")
        if float(self.chitosan_wt_percent) + float(self.cellulose_acetate_wt_percent) + float(self.biochar_wt_percent) > 100:
            raise ValueError("Formulation fractions cannot total more than 100 wt%.")
        if not str(self.membrane_id).strip():
            raise ValueError("membrane_id is required.")

    def to_record(self):
        return asdict(self)

