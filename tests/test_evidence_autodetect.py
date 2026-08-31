from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_eval_appends_view_params_for_dino_view_checkpoints():
    src = (REPO / "scripts/eval_hypernet.py").read_text()
    assert 'checkpoint.get("evidence") == "dino_view"' in src
    assert "append_view_params" in src


def test_export_appends_view_params_for_dino_view_checkpoints():
    src = (REPO / "scripts/export_generated_lora.py").read_text()
    assert 'checkpoint.get("evidence") == "dino_view"' in src
    assert "append_view_params" in src
