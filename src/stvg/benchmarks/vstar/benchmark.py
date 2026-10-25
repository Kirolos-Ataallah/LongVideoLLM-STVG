"""V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning (arXiv:2503.11495).

Each of the 2,094 samples is queried five times (``vqa``, chain 1 ``temporal``/``spatial`` and
chain 2 ``spatial_2``/``temporal_2``). Following the reference protocol, chain queries are
conditioned on *ground-truth* context (GT time span for ``spatial``, GT boxes for
``temporal_2``), so all five queries are independent and can be run in any order.
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from tqdm import tqdm

from stvg.benchmarks.base import Benchmark
from stvg.benchmarks.vstar import parsing
from stvg.benchmarks.vstar.scoring import SPATIAL_THRESHOLDS, aggregate_groups, duration_bucket
from stvg.judges.base import LLMJudge
from stvg.metrics import per_frame_ious, recall_at, temporal_iou, to_pixel_box
from stvg.registry import BENCHMARKS
from stvg.types import Sample, VideoQuery
from stvg.utils.io import JsonlStore, read_json, write_json, write_jsonl
from stvg.utils.logging import get_logger
from stvg.utils.misc import chunked

log = get_logger(__name__)

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")
# Task name -> field name in the reference result JSON.
OFFICIAL_FIELDS = {
    "vqa": "answer_vqa",
    "temporal": "answer_temporal",
    "spatial": "answer_spatial",
    "spatial_2": "answer_spatial_2",
    "temporal_2": "answer_temporal_2",
}


@BENCHMARKS.register("vstar")
class VSTaR(Benchmark):
    tasks = ("vqa", "temporal", "spatial", "spatial_2", "temporal_2")
    requires_judge = True

    def __init__(self, cfg: DictConfig, data_root: str | Path) -> None:
        super().__init__(cfg, data_root)
        self.annotation_path = self.root / cfg.annotation_file
        self.video_dir = self.root / cfg.video_dir
        self.mode = cfg.parsing
        if self.mode not in parsing.MODES:
            raise ValueError(f"parsing must be one of {parsing.MODES}, got {self.mode}")

    # -------------------------------------------------------------------- data
    def prepare(self, force: bool = False) -> None:
        from huggingface_hub import snapshot_download

        raw_dir = self.root / "raw"
        snapshot_download(
            repo_id=self.cfg.hf_repo,
            repo_type="dataset",
            revision=self.cfg.get("hf_revision"),
            local_dir=raw_dir,
        )
        marker = self.video_dir / ".extracted"
        if marker.exists() and not force:
            log.info("Videos already extracted in %s", self.video_dir)
            return
        archive = raw_dir / self.cfg.video_archive
        log.info("Extracting %s -> %s (this takes a while)", archive, self.video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for member in tqdm(zf.infolist(), desc="unzip", unit="file"):
                zf.extract(member, self.video_dir)
        marker.touch()

    def _index_videos(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if self.video_dir.exists():
            for p in self.video_dir.rglob("*"):
                if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._"):
                    index.setdefault(p.stem, p)
        return index

    def load_samples(self) -> list[Sample]:
        annotations = read_json(self.annotation_path)
        videos = self._index_videos()
        samples, missing = [], 0
        for idx, ann in enumerate(annotations):
            path = videos.get(str(ann["vid"]))
            if path is None:
                missing += 1
                continue
            samples.append(Sample(sample_id=f"{idx:05d}", video_path=str(path), annotation=ann))
        if missing:
            msg = f"{missing}/{len(annotations)} V-STaR videos not found under {self.video_dir}"
            if self.cfg.strict_videos:
                raise FileNotFoundError(
                    msg + " (run `stvg prepare vstar` or set strict_videos=false)"
                )
            log.warning(msg + "; those samples are skipped and metrics are NOT comparable.")
        return samples

    # ----------------------------------------------------------------- queries
    @staticmethod
    def _template_fields(ann: dict[str, Any]) -> dict[str, Any]:
        st, et = math.ceil(ann["timestamps"][0]), math.floor(ann["timestamps"][1])
        time_range = list(range(st, et + 1)) or [round(ann["timestamps"][0])]
        gt_boxes = [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in ann["bboxes"]]
        return {
            **ann,
            "video_length": round(ann["frame_count"] / ann["fps"], 2),
            "time_range": time_range,
            "time_range_start": time_range[0],
            "num_bboxes": len(gt_boxes),
            "bboxes": gt_boxes,
        }

    def build_queries(self, sample: Sample) -> list[VideoQuery]:
        fields = self._template_fields(sample.annotation)
        system_prompt = self.cfg.get("system_prompt")
        return [
            VideoQuery(
                qid=self.qid(sample.sample_id, task),
                sample_id=sample.sample_id,
                task=task,
                video_path=sample.video_path,
                prompt=self.cfg.prompts[task].format(**fields).strip(),
                system_prompt=system_prompt,
            )
            for task in self.tasks
        ]

    # ----------------------------------------------------------------- scoring
    def _pixel_boxes(
        self, record: dict[str, Any] | None, ann: dict[str, Any]
    ) -> dict[str, list[list[float]]]:
        if not record or record.get("error"):
            return {}
        raw = parsing.parse_boxes(record.get("text"), self.mode) or {}
        meta = record.get("meta", {})
        fmt, input_hw = meta.get("bbox_format", "abs_original"), meta.get("input_hw")
        return {
            k: [to_pixel_box(b, fmt, ann["width"], ann["height"], input_hw) for b in boxes]
            for k, boxes in raw.items()
        }

    def _text(self, predictions: dict[str, dict], sample_id: str, task: str) -> str | None:
        record = predictions.get(self.qid(sample_id, task))
        if not record or record.get("error"):
            return None
        return record.get("text") or None

    def _judge_vqa(
        self,
        samples: list[Sample],
        predictions: dict[str, dict],
        judge: LLMJudge,
        out_dir: Path,
    ) -> dict[str, int]:
        store = JsonlStore(out_dir / f"judge_{judge.name}.jsonl", key="sample_id")
        jcfg = self.cfg.judge_rubric
        todo = []
        for s in samples:
            answer = self._text(predictions, s.sample_id, "vqa")
            if answer and s.sample_id not in store:
                todo.append((s, answer))
        if todo:
            log.info("Judging %d VQA answers with %s", len(todo), judge.name)
        for batch in tqdm(list(chunked(todo, jcfg.batch_size)), desc="judge", disable=not todo):
            conversations = [
                [
                    {"role": "system", "content": jcfg.system_prompt.strip()},
                    {
                        "role": "user",
                        "content": jcfg.template.format(
                            question=s.annotation["question"],
                            answer=s.annotation["answer"],
                            candidate=answer,
                        ),
                    },
                ]
                for s, answer in batch
            ]
            for (s, _), reply in zip(batch, judge.complete(conversations), strict=True):
                store.add({"sample_id": s.sample_id, "reply": reply})
        return {
            s.sample_id: parsing.parse_judge_score(store.get(s.sample_id)["reply"], self.mode)
            if s.sample_id in store
            else 0  # no answer -> incorrect
            for s in samples
        }

    def evaluate(
        self,
        samples: list[Sample],
        predictions: dict[str, dict[str, Any]],
        judge: LLMJudge | None,
        out_dir: Path,
    ) -> dict[str, Any]:
        vqa_scores = self._judge_vqa(samples, predictions, judge, out_dir) if judge else None
        if judge is None:
            log.warning("No judge configured: VQA accuracy, AM and LGM are not computed.")

        records = []
        for s in samples:
            ann = s.annotation
            gt_boxes = {
                str(b["timestamp"]): [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
                for b in ann["bboxes"]
            }
            rec: dict[str, Any] = {
                "sample_id": s.sample_id,
                "vid": ann["vid"],
                "domain": ann.get("domain", "unknown"),
                "duration": duration_bucket(ann["frame_count"] / ann["fps"]),
                "vqa_score": vqa_scores[s.sample_id] if vqa_scores else None,
            }
            for chain, t_task, s_task in (
                (1, "temporal", "spatial"),
                (2, "temporal_2", "spatial_2"),
            ):
                segment = parsing.parse_timestamps(
                    self._text(predictions, s.sample_id, t_task), self.mode
                )
                boxes = self._pixel_boxes(predictions.get(self.qid(s.sample_id, s_task)), ann)
                ious = per_frame_ious(gt_boxes, boxes)
                rec[f"pred_segment_{chain}"] = segment
                rec[f"tiou_{chain}"] = temporal_iou(ann["timestamps"], segment)
                rec[f"viou_{chain}"] = sum(ious) / len(ious) if ious else 0.0
                rec[f"ap_{chain}"] = recall_at(ious, SPATIAL_THRESHOLDS)
                rec[f"pred_boxes_{chain}"] = boxes
            records.append(rec)

        write_jsonl(records, out_dir / "per_sample.jsonl")
        metrics = aggregate_groups(records, has_vqa=vqa_scores is not None)
        metrics["config"] = {
            "parsing": self.mode,
            "judge": judge.name if judge else None,
            "num_predictions": sum(
                1 for s in samples for t in self.tasks if self._text(predictions, s.sample_id, t)
            ),
            "num_expected_predictions": len(samples) * len(self.tasks),
        }
        return metrics

    def summary_row(self, metrics: dict[str, Any]) -> dict[str, float]:
        o = metrics["overall"]
        row = {"n": o["num_samples"]}
        if "vqa" in o:
            row["Acc"] = o["vqa"]["accuracy"]
        for c in (1, 2):
            row[f"m_tIoU{c}"] = o[f"chain_{c}"]["temporal"]["m_tIoU"]
            row[f"m_vIoU{c}"] = o[f"chain_{c}"]["spatial"]["m_vIoU"]
        if "mAM" in o:
            row["mAM"], row["mLGM"] = o["mAM"], o["mLGM"]
        return row

    # ------------------------------------------------------------------ export
    def export_official(
        self, samples: list[Sample], predictions: dict[str, dict[str, Any]], path: Path
    ) -> None:
        """Write predictions in the reference result format, for cross-checking with V-STaR's
        ``eval.py``. Boxes are converted to original-frame pixels."""
        rows = []
        for s in samples:
            row = dict(s.annotation)
            for task in self.tasks:
                key = OFFICIAL_FIELDS[task]
                text = self._text(predictions, s.sample_id, task)
                if task == "vqa":
                    row[key] = text
                elif task.startswith("temporal"):
                    row[key] = parsing.parse_timestamps(text, self.mode) or []
                else:
                    record = predictions.get(self.qid(s.sample_id, task))
                    boxes = self._pixel_boxes(record, s.annotation)
                    row[key] = {k: v[0] if len(v) == 1 else v for k, v in boxes.items()}
            rows.append(row)
        write_json(rows, path, indent=1)
