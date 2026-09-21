from ui_labels import LABELS


def test_paper_labels_pair_plain_name_with_technical_caption():
    assert LABELS
    for label in LABELS.values():
        assert label["plain"].strip()
        assert label["technical"].strip()
        assert label["plain"] != label["technical"]