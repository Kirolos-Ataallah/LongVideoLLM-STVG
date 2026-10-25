# V-STaR

*V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning* (Cheng et al., 2025,
[arXiv:2503.11495](https://arxiv.org/abs/2503.11495)).
Data: [V-STaR-Bench/V-STaR](https://huggingface.co/datasets/V-STaR-Bench/V-STaR) (CC BY-NC-SA 4.0).

| | |
|---|---|
| Samples | 2,094 QA over 732 videos, 9 domains |
| Video length | mean ~110 s, max ~59 min; buckets short < 60 s ≤ medium < 180 s ≤ long |
| Download | `uv run stvg prepare vstar` (~30 GB zip, extracted to `data/vstar/videos`) |

## Protocol

Each sample is queried five times. The reference code conditions the chained queries on
**ground truth**, so the queries are independent:

| Task | Chain | Asks | Context given |
|---|---|---|---|
| `vqa` | both | *what*: open-ended answer | question |
| `temporal` | 1 (what→when→where) | *when*: `[start, end]` in seconds | video length |
| `spatial` | 1 | *where*: `{second: box}` for each second | GT time span |
| `spatial_2` | 2 (what→where→when) | *where*: `{second: box}` | number of GT boxes |
| `temporal_2` | 2 | *when* | all GT boxes (pixels) + resolution |

## Metrics

- **Accuracy**: an LLM judge rates the VQA answer 0–3 against GT, and ratings ≥ 2 count as correct.
  The official judge is Qwen2.5-72B-Instruct.
- **R1@{0.3,0.5,0.7}** and **m_tIoU**: temporal IoU of the predicted segment.
- **AP@{0.1,…,0.9}** and **m_vIoU**: per sample, the IoU on each annotated frame (missing frame = 0).
  AP@t is the fraction of frames with IoU ≥ t, averaged over samples.
- **AM** = (Acc + m_tIoU + m_vIoU) / 3 and **LGM** = −mean(log(1 − x)) over the same three numbers,
  computed per chain. **mAM** and **mLGM** average the two chains. These are the headline numbers.
- **Combined** success rates: VQA correct (score ≥ 2), tIoU ≥ 0.3, vIoU ≥ 0.1.

Breakdowns are reported overall, by duration bucket and by domain.

## Implementation notes and deviations from the reference code

- **Prompts.** These are copied verbatim from `inference_demo.py` with only the indentation
  whitespace removed. The reference code passes the two temporal prompts as 1-tuples by accident
  (a trailing comma); we pass plain strings.
- **Parsing.** `parsing: official` ports the reference extractors exactly. `parsing: robust` also
  accepts clock times, explicit ranges, `bbox_2d` objects, float second keys and noisy judge
  replies. Robust parsing is only for analysis; report official numbers.
- **Box coordinates.** The reference denormalises boxes from the resized model input
  (Qwen2.5-VL convention). We convert using each model's declared `bbox_format`. For example,
  Qwen3-VL and Qwen3.5 use a relative 0–1000 grid. The chain-2 prompt still gives GT boxes in
  original pixels, exactly as in the reference.
- **Failed generations** (e.g. OOM) count as empty answers and score 0. They are retried
  automatically when inference is resumed.
- **Judge.** We decode greedily for determinism. The judge can run locally (`type: hf`) or behind
  any OpenAI-compatible server (`type: openai`, e.g. vLLM serving Qwen2.5-72B-Instruct). Smaller
  judges such as `qwen2_5_7b_instruct` are fine for development but not comparable with the paper.
- `scoring.export_official: true` writes `official_format.json`, which the reference `eval.py`
  can consume for cross-checking.
