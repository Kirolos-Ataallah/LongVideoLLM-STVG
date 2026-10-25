from stvg.config import load_config
from stvg.registry import BENCHMARKS, JUDGES, MODELS, import_builtin_components


def test_experiment_composes_components():
    cfg = load_config("vstar_qwen3_5_4b")
    assert cfg.model.name == "qwen3_5_4b"
    assert cfg.benchmark.name == "vstar"
    assert cfg.run.name == "vstar_qwen3_5_4b"
    assert cfg.paths.data_root.endswith("data")


def test_overrides_swap_and_edit():
    cfg = load_config(
        "vstar_qwen3_5_4b",
        ["judge=qwen2_5_7b_instruct", "model.generation.max_new_tokens=8", "benchmark.limit=3"],
    )
    assert cfg.judge.name == "qwen2_5_7b_instruct"
    assert cfg.model.generation.max_new_tokens == 8
    assert cfg.benchmark.limit == 3
    assert load_config("vstar_qwen3_5_4b", ["judge=null"]).judge is None


def test_registered_types_match_configs():
    import_builtin_components()
    cfg = load_config("vstar_qwen3_5_4b")
    assert cfg.model.type in MODELS
    assert cfg.benchmark.type in BENCHMARKS
    assert cfg.judge.type in JUDGES
