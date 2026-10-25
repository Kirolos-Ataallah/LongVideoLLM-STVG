from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from omegaconf import DictConfig

from stvg.config import to_plain
from stvg.judges.base import LLMJudge, Messages
from stvg.registry import JUDGES
from stvg.utils.misc import strip_reasoning


@JUDGES.register("openai")
class OpenAICompatJudge(LLMJudge):
    """Any OpenAI-compatible chat endpoint (vLLM / SGLang server, hosted API).

    Example: serve the official V-STaR judge on a multi-GPU node with
    ``vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4`` and point ``base_url`` at it.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        from openai import OpenAI

        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY"),
            timeout=cfg.get("timeout", 120),
            max_retries=cfg.get("max_retries", 5),
        )
        self.gen_kwargs = to_plain(cfg.generation)
        self.extra_body = to_plain(cfg.get("extra_body", {}))
        self.concurrency = int(cfg.get("concurrency", 16))

    def _one(self, messages: Messages) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model_path,
            messages=messages,
            extra_body=self.extra_body or None,
            **self.gen_kwargs,
        )
        return strip_reasoning(resp.choices[0].message.content or "")

    def complete(self, batch: Sequence[Messages]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, batch))
