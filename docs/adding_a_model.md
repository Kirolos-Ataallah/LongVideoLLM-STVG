# Adding a model

A model is anything that maps `(video, prompt)` to text. Baselines and our own models use the
same interface, so every benchmark works with every model.

## 1. Same family, new checkpoint: config only

If the model is already supported by an implementation (e.g. another Qwen3-VL / Qwen3.5
checkpoint), add one YAML file and you are done:

```yaml
# configs/models/qwen3_5_9b.yaml
name: qwen3_5_9b
type: qwen_vl              # registry key of the implementation
model_path: Qwen/Qwen3.5-9B
bbox_format: rel_1000
...                        # copy the rest from qwen3_5_4b.yaml and adjust
```

```bash
uv run stvg run -c vstar_qwen3_5_4b model=qwen3_5_9b run.name=vstar_qwen3_5_9b
```

Or create `configs/experiments/vstar_qwen3_5_9b.yaml` so the run is reproducible by name.

A model config may itself start with `base: <other model config>`. For example,
`qwen3_5_4b_vllm.yaml` inherits everything from `qwen3_5_4b.yaml` and changes only the backend
(`type`) and the engine settings. That guarantees both backends get the same frames and prompts.

## 2. New architecture: one class + one config

Create `src/stvg/models/<my_model>.py`:

```python
from stvg.models.base import VideoLLM
from stvg.registry import MODELS
from stvg.types import ModelOutput, VideoQuery


@MODELS.register("my_model")
class MyModel(VideoLLM):
    def __init__(self, cfg):
        super().__init__(cfg)          # sets self.cfg and self.bbox_format
        import torch                   # heavy imports stay inside __init__
        self.model = ...

    def generate(self, queries: list[VideoQuery]) -> list[ModelOutput]:
        outputs = []
        for q in queries:
            raw = ...                  # load q.video_path, prompt with q.prompt / q.system_prompt
            outputs.append(ModelOutput(
                text=raw,              # final answer (strip any reasoning trace)
                raw_text=raw,
                meta={"bbox_format": str(self.bbox_format),
                      "input_hw": [h, w]},   # only needed for bbox_format=abs_input
            ))
        return outputs

    def close(self):                   # optional: free GPU memory before judging
        del self.model
```

Register the module in `src/stvg/models/__init__.py`:

```python
from stvg.models import my_model  # noqa: F401
```

Then add `configs/models/my_model.yaml` with `type: my_model`.

## Contract

| What | Why |
|---|---|
| Return exactly one `ModelOutput` per query, in order | the runner zips them back to query ids |
| `meta["bbox_format"]` in every output | scoring converts boxes to original-frame pixels (`stvg.metrics.boxes.BoxFormat`) |
| `meta["input_hw"]` when boxes are in resized-input pixels | needed for `abs_input` conversion (Qwen2/2.5-VL style) |
| No benchmark logic inside the model | prompts belong to the benchmark config; frame sampling and chat template belong to the model |
| Exceptions are fine | the runner records the failure, keeps going and retries it on resume |

If the architecture is supported by vLLM, implement it as a shared base class (input
preparation) plus thin backends, as in `qwen_vl.py` (`QwenVLBase`, transformers backend) and
`qwen_vl_vllm.py` (vLLM backend). vLLM backends receive whole batches: set `batch_size` in the
model config so each `generate` call contains many videos.

Frame sampling (`fps`, `max_frames`, pixel budget) is part of the model config because it is
model-specific and directly affects results; keep it explicit so runs are comparable.
