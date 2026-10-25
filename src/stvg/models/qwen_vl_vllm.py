"""Qwen-VL family served by the vLLM offline engine.

Same frames and prompts as :class:`stvg.models.qwen_vl.QwenVL` (shared :class:`QwenVLBase`), but
a whole batch of queries is submitted at once: vLLM schedules them with continuous batching,
caches the vision encoder output per video and reuses the KV cache of shared prefixes
(system prompt + video) across the queries of one video.

Use a large ``batch_size`` in the model config so each ``generate`` call holds many videos.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from omegaconf import DictConfig

from stvg.config import to_plain
from stvg.models.qwen_vl import QwenVLBase, package_version
from stvg.registry import MODELS
from stvg.types import ModelOutput, VideoQuery
from stvg.utils.logging import get_logger

log = get_logger(__name__)

# transformers.generate kwargs -> vLLM SamplingParams, so one `generation` block serves both.
_GEN_KEY_MAP = {"max_new_tokens": "max_tokens"}
_SAMPLING_KEYS = {
    "max_tokens", "temperature", "top_p", "top_k", "min_p", "seed",
    "repetition_penalty", "presence_penalty", "frequency_penalty",
}  # fmt: skip


def sampling_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    out = {_GEN_KEY_MAP.get(k, k): v for k, v in generation.items()}
    if not out.pop("do_sample", True):
        out["temperature"] = 0.0  # greedy
    unknown = set(out) - _SAMPLING_KEYS
    if unknown:
        raise ValueError(f"Unsupported generation keys for vLLM: {sorted(unknown)}")
    return out


@MODELS.register("qwen_vl_vllm")
class QwenVLvLLM(QwenVLBase):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        # vLLM reads its switches from the environment at import time.
        for key, value in to_plain(cfg.get("env", {})).items():
            os.environ.setdefault(key, str(value))
        from vllm import LLM, SamplingParams

        engine = to_plain(cfg.engine)
        log.info("Loading %s from %s (vLLM, %s)", cfg.name, cfg.model_path, engine)
        self.llm = LLM(model=cfg.model_path, dtype=cfg.dtype, seed=cfg.get("seed", 0), **engine)
        self.sampling = SamplingParams(**sampling_kwargs(self.gen_kwargs))
        self.loader_threads = int(cfg.get("video_loader_threads", 8))

    @staticmethod
    def _video_key(q: VideoQuery) -> str:
        return q.video_path + repr(sorted(q.video_kwargs.items()))

    def _load_batch_videos(self, queries: Sequence[VideoQuery]) -> dict[str, tuple[Any, dict]]:
        """Decode every distinct video of the batch once, in parallel (decord releases the GIL)."""
        unique = {self._video_key(q): q for q in queries}
        with ThreadPoolExecutor(max_workers=self.loader_threads) as pool:
            loaded = pool.map(
                lambda q: self.fetch_video(q.video_path, q.video_kwargs), unique.values()
            )
            return dict(zip(unique, loaded, strict=True))

    def generate(self, queries: Sequence[VideoQuery]) -> list[ModelOutput]:
        start = time.perf_counter()
        videos = self._load_batch_videos(queries)
        keys = [self._video_key(q) for q in queries]
        requests = [
            {
                "prompt": self.prompt_text(q),
                "multi_modal_data": {"video": [videos[k]]},
                # Frames are already sampled and resized exactly like the HF backend.
                "mm_processor_kwargs": {"do_sample_frames": False, "do_resize": False},
            }
            for q, k in zip(queries, keys, strict=True)
        ]
        results = self.llm.generate(requests, self.sampling, use_tqdm=False)
        latency = (time.perf_counter() - start) / len(queries)  # amortised over the batch
        outputs = []
        for k, res in zip(keys, results, strict=True):
            video, metadata = videos[k]
            completion = res.outputs[0]
            outputs.append(
                self.make_output(
                    completion.text,
                    video,
                    metadata,
                    len(res.prompt_token_ids),
                    len(completion.token_ids),
                    latency,
                )
            )
        return outputs

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "engine": to_plain(self.cfg.engine),
            "env": to_plain(self.cfg.get("env", {})),
            "backend": {
                "vllm": package_version("vllm"),
                "transformers": package_version("transformers"),
            },
        }

    def close(self) -> None:
        import gc

        import torch

        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
