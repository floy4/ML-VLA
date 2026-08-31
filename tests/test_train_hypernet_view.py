from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_script_uses_view_params_and_marks_checkpoint():
    src = (REPO / "scripts/train_hypernet.py").read_text()
    assert "append_view_params" in src
    assert '"evidence": "dino_view"' in src
    assert '"source": "dino_view"' in src
    assert "onehot" not in src  # no warm-start path
    assert 'default=0.9999' in src
    assert 'default=1e-6' in src


def test_view_configs():
    vw = yaml.safe_load((REPO / "configs/splits_view.yaml").read_text())
    for rung in ("vw1", "vw4"):
        assert vw[rung]["train"] and vw[rung]["test"]
    vwf = yaml.safe_load((REPO / "configs/splits_view_full.yaml").read_text())
    assert vwf["fullvw4"]["train"] and vwf["fullvw4"]["test"]
    cfg = yaml.safe_load((REPO / "configs/hypernet_view.yaml").read_text())
    assert cfg["loss"]["delta_w"] == 0.1
    assert cfg["training"]["max_steps"] == 20000
