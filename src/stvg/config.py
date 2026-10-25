"""Experiment configuration.

An experiment config composes three reusable components, each stored in its own folder::

    configs/
      default.yaml            # global defaults (paths, runtime, inference)
      models/<name>.yaml
      benchmarks/<name>.yaml
      judges/<name>.yaml
      experiments/<name>.yaml # picks components + experiment-specific overrides

A component can be referenced by name, or by name plus overrides::

    model: qwen3_5_4b
    benchmark:
      base: vstar
      limit: 50

Command-line overrides use OmegaConf dot-list syntax. ``model=<name>`` (and likewise for
``benchmark``/``judge``) swaps a whole component; ``model.generation.max_new_tokens=512`` edits
a single field.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

COMPONENTS = {"model": "models", "benchmark": "benchmarks", "judge": "judges"}


def project_root() -> Path:
    return Path(os.environ.get("STVG_ROOT", Path(__file__).resolve().parents[2]))


# ``${root:}`` in YAML expands to the project root, so relative data/output paths do not
# depend on the working directory.
if not OmegaConf.has_resolver("root"):
    OmegaConf.register_resolver("root", lambda: str(project_root()))


def config_root() -> Path:
    return Path(os.environ.get("STVG_CONFIG_ROOT", project_root() / "configs"))


def _resolve_path(ref: str | Path, folder: str) -> Path:
    path = Path(ref)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return path
    candidate = config_root() / folder / f"{ref}.yaml"
    if not candidate.exists():
        available = sorted(p.stem for p in (config_root() / folder).glob("*.yaml"))
        raise FileNotFoundError(f"No config '{ref}' in {folder}/. Available: {available}")
    return candidate


def load_component(kind: str, ref: Any) -> DictConfig | None:
    """Resolve a component reference: a name, a path, or a node with ``base: <name>``.

    ``base`` works both in experiment files and inside component files themselves (e.g. a
    vLLM model config inheriting frame sampling from the HF one), and may be chained.
    """
    if ref is None:
        return None
    if isinstance(ref, str):
        node = OmegaConf.load(_resolve_path(ref, COMPONENTS[kind]))
    else:
        node = ref if isinstance(ref, DictConfig) else OmegaConf.create(ref)
    if "base" not in node:
        return node
    overrides = OmegaConf.masked_copy(node, [k for k in node if k != "base"])
    return OmegaConf.merge(load_component(kind, str(node.base)), overrides)


def load_config(experiment: str | Path, overrides: list[str] | None = None) -> DictConfig:
    overrides = list(overrides or [])
    exp_path = _resolve_path(experiment, "experiments")
    exp = OmegaConf.load(exp_path)

    # Component swaps (``model=foo``) must be applied before the component is resolved.
    field_overrides = []
    for item in overrides:
        key, _, value = item.partition("=")
        if key in COMPONENTS:
            exp[key] = None if value.lower() in {"null", "none", ""} else value
        else:
            field_overrides.append(item)

    for kind in COMPONENTS:
        if kind in exp:
            exp[kind] = load_component(kind, exp[kind])

    cfg = OmegaConf.merge(
        OmegaConf.load(config_root() / "default.yaml"),
        exp,
        OmegaConf.from_dotlist(field_overrides),
    )
    if cfg.run.name is None:
        cfg.run.name = exp_path.stem
    return cfg


def to_plain(node: Any) -> Any:
    """Config node -> plain Python containers (plain values and ``None`` pass through)."""
    if isinstance(node, DictConfig | ListConfig):
        return OmegaConf.to_container(node, resolve=True)
    return node


def run_dir(cfg: DictConfig) -> Path:
    path = Path(cfg.paths.output_root) / cfg.run.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(cfg: DictConfig, path: Path) -> None:
    OmegaConf.save(cfg, path, resolve=True)
