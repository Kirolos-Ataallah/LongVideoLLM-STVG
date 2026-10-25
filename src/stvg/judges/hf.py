from __future__ import annotations

from collections.abc import Sequence

from omegaconf import DictConfig

from stvg.config import to_plain
from stvg.judges.base import LLMJudge, Messages
from stvg.registry import JUDGES
from stvg.utils.logging import get_logger
from stvg.utils.misc import strip_reasoning

log = get_logger(__name__)


@JUDGES.register("hf")
class HFJudge(LLMJudge):
    """A causal LM run locally with transformers (batched, left-padded)."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        log.info("Loading judge %s", cfg.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, dtype=getattr(torch, cfg.dtype), device_map=cfg.device_map
        ).eval()
        self.gen_kwargs = to_plain(cfg.generation)
        self.chat_kwargs = to_plain(cfg.get("chat_template_kwargs", {}))

    def complete(self, batch: Sequence[Messages]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, **self.chat_kwargs
            )
            for m in batch
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs, pad_token_id=self.tokenizer.pad_token_id, **self.gen_kwargs
            )
        texts = self.tokenizer.batch_decode(
            out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return [strip_reasoning(t) for t in texts]

    def close(self) -> None:
        del self.model
        self._torch.cuda.empty_cache()
