"""Model-agnostic evaluation pipeline: inference -> scoring.

Inference writes raw generations to ``<run_dir>/predictions[.shardK-of-N].jsonl`` and resumes
from them, so long runs can be interrupted and split across GPUs (``inference.shard=0/4``, ...).
Scoring only reads those files, so it can be re-run with a different parser or judge without
touching the model.
"""

from __future__ import annotations

import gc
import time
import traceback
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from tqdm import tqdm

from stvg.benchmarks.base import Benchmark
from stvg.config import run_dir, save_config
from stvg.evaluation.report import render_metrics
from stvg.judges.base import LLMJudge
from stvg.models.base import VideoLLM
from stvg.registry import BENCHMARKS, JUDGES, MODELS
from stvg.types import Sample, VideoQuery
from stvg.utils.io import JsonlStore, load_jsonl_shards, write_json
from stvg.utils.logging import get_logger
from stvg.utils.misc import chunked, parse_shard

log = get_logger(__name__)


def build_benchmark(cfg: DictConfig) -> Benchmark:
    return BENCHMARKS.build(cfg.benchmark, data_root=cfg.paths.data_root)


def select_samples(bench: Benchmark, cfg: DictConfig) -> list[Sample]:
    samples = bench.load_samples()
    if cfg.benchmark.get("sample_ids"):
        wanted = {str(x) for x in cfg.benchmark.sample_ids}
        samples = [s for s in samples if s.sample_id in wanted]
    if cfg.benchmark.get("limit"):
        samples = samples[: cfg.benchmark.limit]
    return samples


def _prediction_path(out: Path, shard: tuple[int, int]) -> Path:
    index, total = shard
    suffix = f".shard{index}-of-{total}" if total > 1 else ""
    return out / f"predictions{suffix}.jsonl"


def run_inference(cfg: DictConfig, model: VideoLLM | None = None) -> Path:
    out = run_dir(cfg)
    save_config(cfg, out / "config.yaml")
    bench = build_benchmark(cfg)
    samples = select_samples(bench, cfg)

    tasks = set(cfg.inference.tasks or bench.tasks)
    queries: list[VideoQuery] = [
        q for s in samples for q in bench.build_queries(s) if q.task in tasks
    ]
    # Shard by video so every video is decoded by a single worker.
    shard = parse_shard(cfg.inference.shard)
    videos = sorted({q.video_path for q in queries})
    mine = {v for i, v in enumerate(videos) if i % shard[1] == shard[0]}
    queries = [q for q in queries if q.video_path in mine]

    store = JsonlStore(_prediction_path(out, shard))
    retry = cfg.inference.retry_errors
    todo = [q for q in queries if q.qid not in store or (retry and store.get(q.qid).get("error"))]
    # Group queries of the same video so decoded frames are reused from the model's cache.
    todo.sort(key=lambda q: (q.video_path, q.sample_id, q.task))
    log.info(
        "%s on %s: %d samples, %d queries (%d done, %d to run) -> %s",
        cfg.model.name,
        bench.name,
        len(samples),
        len(queries),
        len(queries) - len(todo),
        len(todo),
        store.path,
    )
    if not todo:
        return out

    model = model or MODELS.build(cfg.model)
    write_json(model.describe(), out / "model.json")
    batch_size = cfg.inference.batch_size or cfg.model.get("batch_size", 1)
    progress = tqdm(total=len(todo), desc=f"infer:{cfg.model.name}", unit="q")
    for batch in chunked(todo, batch_size):
        try:
            outputs = model.generate(batch)
            records = [
                {
                    "qid": q.qid,
                    "sample_id": q.sample_id,
                    "task": q.task,
                    "prompt": q.prompt,
                    **o.to_dict(),
                    "error": None,
                }
                for q, o in zip(batch, outputs, strict=True)
            ]
        except Exception as e:  # keep going; failures are recorded and retried on resume
            if not cfg.inference.continue_on_error:
                raise
            log.error("Batch failed (%s): %s", [q.qid for q in batch], e)
            log.debug(traceback.format_exc())
            _free_cuda()
            records = [
                {
                    "qid": q.qid,
                    "sample_id": q.sample_id,
                    "task": q.task,
                    "prompt": q.prompt,
                    "text": "",
                    "raw_text": "",
                    "meta": {},
                    "error": f"{type(e).__name__}: {e}",
                }
                for q in batch
            ]
        for r in records:
            store.add(r)
        progress.update(len(batch))
    progress.close()
    model.close()
    del model
    _free_cuda()
    return out


def run_scoring(cfg: DictConfig) -> dict[str, Any]:
    out = run_dir(cfg)
    bench = build_benchmark(cfg)
    samples = select_samples(bench, cfg)
    predictions = load_jsonl_shards(out.glob("predictions*.jsonl"))
    if not predictions:
        raise FileNotFoundError(f"No predictions in {out}; run `stvg infer` first.")
    errors = sum(1 for r in predictions.values() if r.get("error"))
    if errors:
        log.warning("%d predictions are failed generations and score as empty answers", errors)

    judge: LLMJudge | None = None
    if bench.requires_judge and cfg.get("judge") is not None:
        judge = JUDGES.build(cfg.judge)

    start = time.perf_counter()
    metrics = bench.evaluate(samples, predictions, judge, out)
    if judge is not None:
        judge.close()
    metrics["meta"] = {
        "run": cfg.run.name,
        "model": cfg.model.name,
        "benchmark": bench.name,
        "num_failed_generations": errors,
        "scoring_time_s": round(time.perf_counter() - start, 1),
    }
    metrics["summary"] = bench.summary_row(metrics)
    write_json(metrics, out / "metrics.json")
    (out / "report.md").write_text(render_metrics(metrics), encoding="utf-8")
    if cfg.scoring.get("export_official") and hasattr(bench, "export_official"):
        bench.export_official(samples, predictions, out / "official_format.json")
    log.info("Wrote %s", out / "metrics.json")
    return metrics


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
