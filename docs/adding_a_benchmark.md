# Adding a benchmark

A benchmark owns its data, its prompts, how answers are parsed and how metrics are computed. It
never touches the model directly: it emits `VideoQuery` objects and later scores the stored
answers.

## Steps

1. Create a package `src/stvg/benchmarks/<name>/` with a class implementing
   `stvg.benchmarks.base.Benchmark`:

   ```python
   @BENCHMARKS.register("my_bench")
   class MyBench(Benchmark):
       tasks = ("vqa", "grounding")      # one query per task per sample
       requires_judge = False            # True if open-ended answers need an LLM judge

       def prepare(self, force=False):   # download / extract into self.root (idempotent)
           ...

       def load_samples(self) -> list[Sample]:
           ...                           # stable sample_id, absolute video_path, raw annotation

       def build_queries(self, sample) -> list[VideoQuery]:
           ...                           # prompts rendered from self.cfg.prompts[task]

       def evaluate(self, samples, predictions, judge, out_dir) -> dict:
           ...                           # parse predictions[qid]["text"], compute metrics,
                                         # write per-sample details to out_dir

       def summary_row(self, metrics) -> dict:
           ...                           # headline numbers for `stvg report`
   ```

2. Import it in `src/stvg/benchmarks/__init__.py` so it registers.
3. Add `configs/benchmarks/<name>.yaml` with `type: my_bench`, data locations and prompt templates.
4. Add an experiment config, e.g. `configs/experiments/<name>_qwen3_5_4b.yaml`.
5. Add tests for parsing and metrics, ideally checked against the official evaluation code
   (see `tests/reference_vstar.py` and `tests/test_metrics.py`).

## Guidelines

- **Reuse generic metrics.** `stvg.metrics` provides temporal IoU, box IoU, per-frame IoU,
  recall@IoU and box-format conversion. Add new generic metrics there and keep only
  benchmark-specific aggregation in the benchmark package.
- **Convert boxes with the model's declared format.** Every prediction record carries
  `meta.bbox_format` (and `meta.input_hw`). Use `stvg.metrics.to_pixel_box` instead of assuming
  a coordinate system.
- **Prompts live in YAML.** Use Python `str.format` placeholders; escape literal braces as `{{ }}`.
- **Official protocol first.** Default to the reference prompts, parsing and metrics so numbers
  are comparable with papers. Put any deviation behind a config flag (see V-STaR's
  `parsing: official | robust`) and document it in `docs/benchmarks/<name>.md`.
- **Keep queries independent** when the protocol allows it; the runner then reorders them by
  video for cache reuse and shards them across GPUs.
