# CKD Deepfake

**Continual Knowledge Distillation for Cross-Generational Deepfake Detection on Edge Devices**

Master's thesis, S2 Teknik Elektro ITB.

## Overview

CKD combines three directions to build a compact deepfake detector that keeps
up with new generations of synthesis techniques without retraining from
scratch:

1. **Knowledge Distillation** from an ensemble of large teachers
   (EfficientNet-B4, RECCE, CLIP ViT-L/14, ~476M params total)
   into a small student (MobileNetV3-Large, ~5.4M params).
2. **Continual Learning** with anti-forgetting mechanisms
   (EWC, LwF, replay buffer) as new generations of deepfakes appear.
3. **Edge Deployment** via TFLite (float16 / int8) for on-device inference.

## Generations

All three generations are drawn from the [DF40 dataset](https://github.com/YZY-stack/DF40)
(40 pre-cropped face techniques spanning the full deepfake history).
DF40 ships face-aligned 224×224 crops directly, so no per-video face extraction is required.

| Gen  | DF40 category                       | Example techniques                                 |
|------|-------------------------------------|----------------------------------------------------|
| Gen1 | Face-Swap (FS, classic)             | faceswap, fsgan, simswap, inswap, blendface, …    |
| Gen2 | Face-Reenactment (FR, talking-head) | fomm, facevid2vid, wav2lip, sadtalker, heygen, …  |
| Gen3 | Entire-Face Synthesis + diffusion   | sd2.1, ddim, DiT, SiT, PixArt, MidJourney, …      |

See `configs/default.yaml -> data.generations` for the full per-gen technique list.

## Pipeline

```
01b_catalog_df40         # DF40 image tree -> metadata CSV (per generation)
02_generate_splits       # 70/15/15 video-level splits
03_generate_soft_labels  # teacher ensemble -> .npy
04_initial_distillation  # student on Gen1
05_continual_distillation# student on Gen2, Gen3 with anti-forgetting
06_ablation_study        # A1..A5 sensitivity
07_edge_evaluation       # TFLite conversion + latency benchmark
08_generate_figures      # thesis figures / tables
```

## Project layout

```
configs/       # YAML configs
src/
  data/        # face extraction, dataset, augmentation, splits
  models/
    teachers/  # EfficientNet-B4, RECCE, CLIP, ensemble
    students/  # MobileNetV3
  training/
    anti_forgetting/  # EWC, LwF, replay
  evaluation/  # metrics, evaluator, edge_eval
  utils/       # config, logger, checkpoint, colab utils
scripts/       # 01..08 pipeline entrypoints
tests/         # pytest unit tests
notebooks/     # Colab setup notebook
```

## Setup

### Local (development)

```bash
pip install -r requirements.txt
pip install -e .
pytest tests/
```

### Colab (training)

Open `notebooks/colab_run_all.ipynb` and *Runtime → Run all*.
Requires Colab Pro (A100 recommended) and a Google Drive folder at
`/content/drive/MyDrive/CKD_Thesis/`. The notebook downloads DF40
(~55 GB) and teacher checkpoints on first run, then executes the full
pipeline (catalog → splits → soft labels → initial & continual
distillation → edge eval → figures).

## New metrics

- **CDE** — Continual Distillation Efficiency (AUC vs. latency vs. size)
- **CGRS** — Cross-Generation Retention Score (AUC on old gens after updates)

See `src/evaluation/metrics.py` and `tests/test_metrics.py`.

## Status

Phase 1 (project scaffolding) complete. See `CKD_Implementation_Plan.md`
in the parent directory for the full roadmap.
