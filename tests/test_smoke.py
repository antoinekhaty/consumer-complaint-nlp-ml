import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import FEATURES, build_pipeline, clean_text, make_demo_frame, run


def test_synthetic_data_and_pipeline():
    frame = make_demo_frame()
    assert len(frame) == 240
    assert frame["company_response"].nunique() == 6
    assert set(FEATURES) == {"narrative", "product", "issue", "state"}
    assert build_pipeline().named_steps["features"].remainder == "drop"
    assert clean_text("  Credit!!!  Fee  ") == "credit fee"


def test_training_smoke():
    metrics = run(make_demo_frame())
    assert metrics["test_n"] == 48
    for key in ("test_weighted_f1", "test_macro_f1", "majority_weighted_f1"):
        assert 0 <= metrics[key] <= 1
