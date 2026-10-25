"""Qwen-VL family (Qwen3-VL, Qwen3.5): shared preprocessing + HuggingFace transformers backend.

Frames are sampled with ``qwen-vl-utils`` so that the sampling budget (fps / max_frames /
total_pixels) is explicit in the config, and the processor receives the true frame indices to
render absolute ``<t seconds>`` timestamps, which matters for temporal grounding.

:class:`QwenVLBase` holds everything that defines *what the model sees* (frame sampling, chat
template, prompt layout). Backends (:class:`QwenVL` here, ``QwenVLvLLM`` in ``qwen_vl_vllm.py``)
only differ in *how* generation runs, so their results are directly comparable.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

from stvg.config import to_plain
from stvg.models.base import VideoLLM
from stvg.registry import MODELS
from stvg.types import ModelOutput, VideoQuery
from stvg.utils.logging import get_logger
from stvg.utils.misc import strip_reasoning

log = get_logger(__name__)


def package_version(name: str) -> str | None:
    """Installed version of an optional package, ``None`` if absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class QwenVLBase(VideoLLM):
    """Backend-independent Qwen-VL logic: video loading, prompt construction, output meta."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(cfg.model_path)
        # Pin the decoder: otherwise qwen-vl-utils picks whichever reader happens to be installed
        # (torchcodec > decord > torchvision), and different decoders yield different frames.
        import qwen_vl_utils.vision_process as qwen_vision

        qwen_vision.FORCE_QWENVL_VIDEO_READER = cfg.get("video_reader", "decord")
        self.video_cfg: dict[str, Any] = to_plain(cfg.video)
        self.gen_kwargs: dict[str, Any] = to_plain(cfg.generation)
        self.chat_kwargs: dict[str, Any] = to_plain(cfg.get("chat_template_kwargs", {}))
        # Benchmarks issue several queries per video; keep the decoded frames of the last few.
        self._video_cache: OrderedDict[str, tuple[Any, dict]] = OrderedDict()
        self._video_cache_size = int(cfg.get("video_cache_size", 2))

    def fetch_video(self, path: str, overrides: dict[str, Any]) -> tuple[Any, dict]:
        """Sampled + resized frames ``(T, C, H, W)`` and qwen-vl-utils metadata (uncached)."""
        from qwen_vl_utils import fetch_video

        element = {"type": "video", "video": path, **self.video_cfg, **overrides}
        return fetch_video(
            element, image_patch_size=self.cfg.patch_size, return_video_metadata=True
        )

    def load_video(self, path: str, overrides: dict[str, Any]) -> tuple[Any, dict]:
        """:meth:`fetch_video` behind a small LRU cache (not thread-safe)."""
        key = json.dumps([path, overrides], sort_keys=True, default=str)
        if key in self._video_cache:
            self._video_cache.move_to_end(key)
            return self._video_cache[key]
        video, metadata = self.fetch_video(path, overrides)
        self._video_cache[key] = (video, metadata)
        while len(self._video_cache) > self._video_cache_size:
            self._video_cache.popitem(last=False)
        return video, metadata

    def prompt_text(self, query: VideoQuery) -> str:
        """Chat-formatted prompt with a single video placeholder."""
        video_item = {"type": "video", "video": query.video_path}
        text_item = {"type": "text", "text": query.prompt}
        content = (
            [video_item, text_item]
            if self.cfg.get("video_first", True)
            else [text_item, video_item]
        )
        messages = []
        system_prompt = query.system_prompt or self.cfg.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.chat_kwargs
        )

    def make_output(
        self, raw: str, video: Any, metadata: dict, num_input: int, num_output: int, latency: float
    ) -> ModelOutput:
        num_frames, _, height, width = video.shape
        duration = metadata["total_num_frames"] / metadata["fps"]
        return ModelOutput(
            text=strip_reasoning(raw),
            raw_text=raw,
            meta={
                "bbox_format": str(self.bbox_format),
                "input_hw": [int(height), int(width)],
                "num_frames": int(num_frames),
                "sample_fps": len(metadata["frames_indices"]) / max(duration, 1e-6),
                "num_input_tokens": int(num_input),
                "num_output_tokens": int(num_output),
                "latency_s": round(latency, 3),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "type": self.cfg.type,
            "model_path": self.cfg.model_path,
            "video": self.video_cfg,
            "video_reader": self.cfg.get("video_reader", "decord"),
            "generation": self.gen_kwargs,
            "chat_template_kwargs": self.chat_kwargs,
        }


@MODELS.register("qwen_vl")
class QwenVL(QwenVLBase):
    """One query at a time with ``transformers.generate``; the reference / development backend."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForImageTextToText

        self._torch = torch
        log.info("Loading %s from %s (transformers)", cfg.name, cfg.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_path,
            dtype=getattr(torch, cfg.dtype),
            attn_implementation=cfg.attn_implementation,
            device_map=cfg.device_map,
        ).eval()

    def _generate_one(self, query: VideoQuery) -> ModelOutput:
        start = time.perf_counter()
        video, metadata = self.load_video(query.video_path, query.video_kwargs)
        inputs = self.processor(
            text=[self.prompt_text(query)],
            videos=[video],
            video_metadata=[metadata],
            do_resize=False,
            do_sample_frames=False,
            return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **self.gen_kwargs)
        new_tokens = output_ids[0, inputs.input_ids.shape[1] :]
        raw = self.processor.decode(new_tokens, skip_special_tokens=True)
        return self.make_output(
            raw,
            video,
            metadata,
            inputs.input_ids.shape[1],
            new_tokens.shape[0],
            time.perf_counter() - start,
        )

    def generate(self, queries: Sequence[VideoQuery]) -> list[ModelOutput]:
        return [self._generate_one(q) for q in queries]

    def describe(self) -> dict[str, Any]:
        # Kernel backends change bf16 numerics, hence greedy outputs: record them.
        return {
            **super().describe(),
            "backend": {
                "transformers": package_version("transformers"),
                "flash_linear_attention": package_version("flash-linear-attention"),
                "causal_conv1d": package_version("causal-conv1d"),
            },
        }

    def close(self) -> None:
        self._video_cache.clear()
        del self.model
        self._torch.cuda.empty_cache()
