# LongVideoLLM-STVG

Research codebase for **long video understanding with spatio-temporally grounded answers**: a
Video-LLM should say *what* happens, *when* it happens (time span), and *where* it happens
(bounding boxes).

The repository holds our own model development and a unified evaluation harness for baselines.
Every model runs on every benchmark through the same pipeline, configured entirely by YAML.

| Benchmarks | Models |
|---|---|
| [V-STaR](docs/benchmarks/vstar.md) — what / when / where reasoning chains | Qwen3.5-4B: `qwen3_5_4b_vllm` (vLLM), `qwen3_5_4b` (transformers) |

### Inference backends

| Backend | Model `type` | Use it for | 10 V-STaR samples* |
|---|---|---|---|
| vLLM | `qwen_vl_vllm` | baseline evaluation: batched, deterministic | 36 s |
| transformers | `qwen_vl` | debugging and our own model development (hooks, custom forward) | 197 s |

\*Inference time on one RTX PRO 4500 Blackwell (32 GB), short videos, excluding model loading.

Both backends share frame sampling, the video reader and the chat template. On V-STaR they
produce identical input token sequences. Greedy outputs still differ slightly because the kernels
round differently in bf16, so only compare numbers obtained with the same backend.

**Reproducibility.** The vLLM config disables prefix caching. With it enabled, inference is
about 1.8x faster, but only about half of the greedy answers were identical across two identical
runs, because cache reuse depends on scheduling timing. With it disabled, repeated runs are
bit-identical. For quick exploratory runs, add `model.engine.enable_prefix_caching=true`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs its own Python 3.12, which ships the
headers Triton needs, so no system Python packages are involved.

One-time system setup (Ubuntu 24.04):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                        # uv
sudo apt install build-essential                                         # C compiler (Triton)
# CUDA toolkit 13.0 = torch's CUDA version (toolkit only; keep the existing NVIDIA driver)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
echo 'export CUDA_HOME=/usr/local/cuda-13.0; export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
```

Then:

```bash
uv sync --all-extras        # creates .venv; compiles causal-conv1d once (~5 min)
uv run pytest               # sanity check
```

Extras:
- `hf`: inference with transformers.
- `vllm`: the vLLM engine (`vllm==0.32`). It pins torch 2.13 and transformers < 5.20.
- `fast`: kernels for Qwen3.5 linear attention in the transformers backend
  (`flash-linear-attention`, plus `causal-conv1d`, which is compiled locally). This extra needs
  the CUDA toolkit. On a machine without it, use `uv sync --extra hf --extra vllm`. Without the
  kernels, transformers falls back to a slower pure-torch path.

Always sync with `--all-extras`, or list every extra: `uv sync` removes extras you leave out.

Evaluation does not need the CUDA toolkit. The vLLM config keeps FlashInfer's JIT-compiled
sampler off (`env.VLLM_USE_FLASHINFER_SAMPLER: 0`), which made no speed difference for greedy
decoding. The video decoder is pinned to `decord` (`video_reader`) because `torchcodec` would
need system FFmpeg. Each run's `model.json` records the backend, the package versions and these
settings.

## Quickstart

```bash
uv run stvg prepare vstar                              # download + unzip (~30 GB)
uv run stvg run -c vstar_qwen3_5_4b_vllm               # inference + scoring (vLLM)
```

Common variations:

```bash
# Quick debug run on 10 samples, in a separate output directory
uv run stvg run -c vstar_qwen3_5_4b_vllm benchmark.limit=10 run.name=debug

# Same experiment on the transformers backend
uv run stvg run -c vstar_qwen3_5_4b

# Inference only (resumable: re-running continues where it stopped)
uv run stvg infer -c vstar_qwen3_5_4b_vllm

# Multi-GPU: one process per GPU, sharded by video, then score
scripts/infer_multi_gpu.sh vstar_qwen3_5_4b_vllm 4

# Re-score without the model: other judge, robust answer parsing
uv run stvg score -c vstar_qwen3_5_4b_vllm judge=qwen2_5_7b_instruct benchmark.parsing=robust

# Compare runs
uv run stvg report outputs/*

# Inspect the fully resolved config / list available components
uv run stvg show -c vstar_qwen3_5_4b
uv run stvg list
```

**VQA judge.** The official V-STaR judge is Qwen2.5-72B-Instruct, which does not fit on a
single 32 GB GPU. Serve it on a larger node, e.g.
`vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4`, and set
`STVG_JUDGE_URL=http://<host>:8000/v1`. For local development use
`judge=qwen2_5_7b_instruct`, but those numbers are not paper-comparable. With `judge=null` you
still get all grounding metrics. Only Accuracy, AM and LGM are skipped.

## Project layout

```
configs/
  default.yaml               global defaults: paths, runtime, inference options
  models/                    one file per model (checkpoint, frame sampling, generation, bbox format)
  benchmarks/                one file per benchmark (data location, prompts, parsing, judge rubric)
  judges/                    LLM judges for open-ended answers
  experiments/               model + benchmark + judge (+ overrides) = one reproducible run
src/stvg/
  cli.py                     `stvg` command: prepare | infer | score | run | report | show | list
  config.py                  config composition and overrides
  registry.py                MODELS / BENCHMARKS / JUDGES registries
  types.py                   Sample, VideoQuery, ModelOutput
  models/                    VideoLLM interface + implementations (baselines and ours)
  benchmarks/                Benchmark interface + implementations (data, prompts, parsing, scoring)
  judges/                    LLMJudge interface: local HF or OpenAI-compatible endpoint
  metrics/                   generic grounding metrics: temporal IoU, box IoU, box formats
  evaluation/                runner (resumable/sharded inference, scoring) and reports
  utils/
tests/                       unit tests, incl. parity with the official V-STaR metric code
docs/                        how-tos and per-benchmark protocol notes
scripts/                     launch helpers
data/        (git-ignored)   benchmarks, set STVG_DATA_ROOT to relocate
outputs/     (git-ignored)   runs, set STVG_OUTPUT_ROOT to relocate
```

## Configuration

An experiment names its components and may override any field:

```yaml
# configs/experiments/vstar_qwen3_5_4b.yaml
model: qwen3_5_4b
benchmark: vstar
judge: qwen2_5_72b_instruct
```

```yaml
# a component can also be referenced with overrides
model:
  base: qwen3_5_4b
  chat_template_kwargs: {enable_thinking: true}
```

Command-line overrides use dot-list syntax. `model=<name>` swaps a component.
`model.video.max_frames=256` edits a field. The resolved config is saved with every run.

## Outputs

```
outputs/<run.name>/
  config.yaml                resolved config
  model.json                 model description (checkpoint, sampling, generation)
  predictions*.jsonl         one record per query: prompt, raw/final text, meta (frames, tokens, latency)
  judge_<judge>.jsonl        cached judge replies
  per_sample.jsonl           parsed predictions + per-sample metrics
  metrics.json               all metrics (overall, by duration, by domain)
  report.md                  readable tables
  official_format.json       predictions in the benchmark's reference format
  logs/
```

## Extending

- New model: [docs/adding_a_model.md](docs/adding_a_model.md). A new checkpoint of a supported
  family is a single YAML file.
- New benchmark: [docs/adding_a_benchmark.md](docs/adding_a_benchmark.md).

Our own models implement the same `VideoLLM` interface in `src/stvg/models/`, so they are
evaluated with exactly the same pipeline as the baselines.

## Development

```bash
uv run pytest
uv run ruff check src tests && uv run ruff format src tests
uv run pre-commit install
```
