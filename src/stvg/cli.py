"""Command-line entry point: ``stvg <command> ...`` (installed by ``uv sync``).

Examples::

    stvg prepare vstar
    stvg run   -c vstar_qwen3_5_4b
    stvg infer -c vstar_qwen3_5_4b benchmark.limit=20 run.name=debug
    stvg infer -c vstar_qwen3_5_4b inference.shard=0/2      # one process per GPU
    stvg score -c vstar_qwen3_5_4b judge=qwen2_5_7b_instruct benchmark.parsing=robust
    stvg report outputs/*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from stvg.config import config_root, load_component, load_config, run_dir
from stvg.registry import BENCHMARKS, JUDGES, MODELS, import_builtin_components
from stvg.utils.logging import setup_logging
from stvg.utils.misc import set_seed


def _experiment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", required=True, help="experiment name or path to YAML")
    p.add_argument("overrides", nargs="*", help="dot-list overrides, e.g. model.dtype=float16")


def _setup(args: argparse.Namespace, stage: str):
    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.run.log_level, run_dir(cfg) / "logs" / f"{stage}.log")
    set_seed(cfg.run.seed)
    return cfg


def cmd_infer(args: argparse.Namespace) -> None:
    from stvg.evaluation import run_inference

    run_inference(_setup(args, "infer"))


def cmd_score(args: argparse.Namespace) -> None:
    from stvg.evaluation import run_scoring

    metrics = run_scoring(_setup(args, "score"))
    print(OmegaConf.to_yaml(OmegaConf.create(metrics["summary"])))


def cmd_run(args: argparse.Namespace) -> None:
    from stvg.evaluation import run_inference, run_scoring

    cfg = _setup(args, "run")
    run_inference(cfg)
    metrics = run_scoring(cfg)
    print(OmegaConf.to_yaml(OmegaConf.create(metrics["summary"])))


def cmd_prepare(args: argparse.Namespace) -> None:
    setup_logging()
    cfg = load_component("benchmark", args.benchmark)
    defaults = OmegaConf.load(config_root() / "default.yaml")
    BENCHMARKS.build(cfg, data_root=defaults.paths.data_root).prepare(force=args.force)


def cmd_report(args: argparse.Namespace) -> None:
    from stvg.evaluation.report import compare_runs

    print(compare_runs(Path(p) for p in args.runs))


def cmd_show(args: argparse.Namespace) -> None:
    print(OmegaConf.to_yaml(load_config(args.config, args.overrides), resolve=True))


def cmd_list(_: argparse.Namespace) -> None:
    for kind, registry in (("models", MODELS), ("benchmarks", BENCHMARKS), ("judges", JUDGES)):
        configs = sorted(p.stem for p in (config_root() / kind).glob("*.yaml"))
        print(f"{kind}:\n  types:   {registry.keys()}\n  configs: {configs}")
    print(f"experiments: {sorted(p.stem for p in (config_root() / 'experiments').glob('*.yaml'))}")


def main(argv: list[str] | None = None) -> None:
    import_builtin_components()
    parser = argparse.ArgumentParser(prog="stvg", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_ in (
        ("infer", cmd_infer, "run a model on a benchmark (resumable)"),
        ("score", cmd_score, "score stored predictions"),
        ("run", cmd_run, "infer + score"),
        ("show", cmd_show, "print the resolved experiment config"),
    ):
        p = sub.add_parser(name, help=help_)
        _experiment_args(p)
        p.set_defaults(fn=fn)

    p = sub.add_parser("prepare", help="download and extract a benchmark")
    p.add_argument("benchmark", help="benchmark config name, e.g. vstar")
    p.add_argument("--force", action="store_true", help="re-extract videos")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("report", help="compare runs")
    p.add_argument("runs", nargs="+", help="run directories")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("list", help="list registered components and configs")
    p.set_defaults(fn=cmd_list)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
