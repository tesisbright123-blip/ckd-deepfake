# Edge Evaluation Guide — MacBook M2 Max

Step-by-step instructions for running the CKD edge deployment evaluation
on a MacBook Apple Silicon machine.

**Scope:** Convert 9 student checkpoints to TFLite (fp32 + int8) and
CoreML (fp32 + int8), benchmark latency, evaluate AUC on the 3
generation test splits (gen3 full + gen1/gen2 5K subset).

**Estimated time:** 3–5 hours wall-clock on M2 Max.

---

## 0. Prerequisites

You will need:

- **MacBook with Apple Silicon** (M1/M2/M3/M4 chip)
  - Intel MacBooks will run TFLite via x86 but cannot benchmark CoreML on ANE
- **macOS 13 (Ventura) or newer** for full CoreML feature support
- **Python 3.10 or 3.11** (3.12+ has not been validated with all wheels)
- **At least 10 GB free disk space** (1.2 GB data + env + intermediate files)
- **Internet** for one-time package install
- **The 9 student checkpoints** (download from Drive — see Step 1.3)
- **DF40 ZIP archives** (download from Drive — see Step 1.3)
- **Split CSV files** (download from Drive — see Step 1.3)

---

## 1. Download Required Artifacts

The MacBook workflow is **offline-by-design** — there is no Drive mount
or `gdown` integration. You download the files once manually via the
Drive web UI, then the script handles the rest.

### 1.1 Where to find the files on Drive

Sign in to `arrantisi.online@gmail.com` Drive and navigate to
`My Drive/CKD_Thesis/`. Required folders:

| Folder | Contents | Total |
|---|---|---|
| `datasets/raw/df40_zip_backup/` | 29 ZIPs (9 gen1 + 11 gen2 + 9 gen3) | ~45 GB |
| `datasets/raw/df40_zip_backup/` (continued) | 2 real ZIPs (FF++, Celeb-DF) | ~3 GB |
| `datasets/splits/` | `gen{1,2,3}_test.csv` | ~30 MB |
| `checkpoints/students/` | 9 student checkpoints | ~820 MB |

### 1.2 Suggested local layout

Place downloaded files anywhere — the setup script takes 3 separate path
flags so you can put them where you like. A simple layout:

```
~/Downloads/
├── df40_zips/                    <- ZIPs go here
│   ├── blendface.zip
│   ├── e4s.zip
│   ├── ...
│   ├── FaceForensics++_real_data_for_DF40.zip
│   └── Celeb-DF-v2_real_data_for_DF40.zip
├── ckd_ckpts/                    <- Checkpoint folders go here
│   ├── gen1_seed0/
│   │   └── best.pth
│   ├── gen1_seed1/
│   │   └── best.pth
│   ├── ...
│   └── gen3_replay+ewc_seed2/
│       └── best.pth
└── ckd_splits/                   <- Test CSVs go here
    ├── gen1_test.csv
    ├── gen2_test.csv
    └── gen3_test.csv
```

### 1.3 ZIP files you actually need

For full evaluation (gen3 full + gen1/gen2 cross-gen), download ALL of
these. You can skip ZIPs whose techniques never appear in the test
CSVs but it's simpler to just download everything.

**Gen1 (face-swap):** `blendface.zip`, `e4s.zip`, `facedancer.zip`,
`faceswap.zip`, `fsgan.zip`, `inswap.zip`, `mobileswap.zip`,
`simswap.zip`, `uniface.zip`

**Gen2 (reenactment):** `fomm.zip`, `facevid2vid.zip`, `wav2lip.zip`,
`MRAA.zip`, `one_shot_free.zip`, `pirender.zip`, `tpsm.zip`, `lia.zip`,
`danet.zip`, `sadtalker.zip`, `mcnet.zip`

**Gen3 (diffusion):** `sd2.1.zip`, `ddim.zip`, `pixart.zip`, `DiT.zip`,
`SiT.zip`, `StyleGAN2.zip`, `StyleGAN3.zip`, `StyleGANXL.zip`, `VQGAN.zip`

**Real:** `FaceForensics++_real_data_for_DF40.zip`,
`Celeb-DF-v2_real_data_for_DF40.zip`

### 1.4 Checkpoint files needed

All 9 of these (each is a folder with `best.pth` inside):

- `gen1_seed0/best.pth`
- `gen1_seed1/best.pth`
- `gen1_seed2/best.pth`
- `gen2_replay+ewc_seed0/best.pth`
- `gen2_replay+ewc_seed1/best.pth`
- `gen2_replay+ewc_seed2/best.pth`
- `gen3_replay+ewc_seed0/best.pth`
- `gen3_replay+ewc_seed1/best.pth`
- `gen3_replay+ewc_seed2/best.pth`

---

## 2. Install the Python Environment

### 2.1 Clone the repository

```bash
cd ~
git clone https://github.com/tesisbright123-blip/ckd-deepfake.git
cd ckd-deepfake
```

### 2.2 Pick a package manager

`uv` is faster but less battle-tested; conda has broader Apple Silicon
wheel support. Both work.

#### Option A: uv (recommended for speed)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv with Python 3.11
uv venv .venv --python 3.11
source .venv/bin/activate

# Install MacBook-specific deps
uv pip install -r requirements-macbook.txt

# Editable install of the repo
uv pip install -e .
```

#### Option B: conda (miniforge)

```bash
# Install miniforge if you don't have it: https://github.com/conda-forge/miniforge
conda create -n ckd-edge python=3.11
conda activate ckd-edge

pip install -r requirements-macbook.txt
pip install -e .
```

### 2.3 Sanity check the install

```bash
python -c "
import torch, onnx, onnx2tf, coremltools
print('torch       :', torch.__version__)
print('onnx        :', onnx.__version__)
print('onnx2tf     :', onnx2tf.__version__ if hasattr(onnx2tf, '__version__') else 'unknown')
print('coremltools :', coremltools.__version__)
import platform
print('macOS arm64 :', platform.system() == 'Darwin' and platform.machine() == 'arm64')
"
```

Expected output includes `coremltools : 7.x.x` and `macOS arm64 : True`.

---

## 3. Run the Setup Script

This extracts ZIPs to local disk, copies the 9 checkpoints into the
expected layout, and rewrites CSV paths so they point to your MacBook
filesystem instead of the original Colab NVMe paths.

```bash
python scripts/edge/setup_macbook_mirror.py \
    --zip-dir   ~/Downloads/df40_zips \
    --ckpt-dir  ~/Downloads/ckd_ckpts \
    --csv-dir   ~/Downloads/ckd_splits \
    --output-root ~/ckd-edge
```

Add `--resume` if you re-run after a partial failure — already-extracted
ZIPs and already-copied checkpoints are skipped.

### Expected output

```
INFO  Step 1/4: extracting 29 ZIPs from ~/Downloads/df40_zips -> ~/ckd-edge
INFO    [ext  ] blendface.zip (1.8 GB) -> ~/ckd-edge/df40
INFO    [ext  ] e4s.zip (1.8 GB) -> ~/ckd-edge/df40
...
INFO  Step 2/4: copying 9 student checkpoints -> ~/ckd-edge/mirror/checkpoints/students
INFO    [copy ] gen1_seed0/best.pth -> ... (54.7 MB)
...
INFO  Step 3/4: rewriting 3 test CSVs -> ~/ckd-edge/mirror/datasets/splits
INFO    [csv  ] gen1_test.csv -> ... (39968 rows, sampled 50/50 exist on disk)
...
INFO  Step 4/4: wrote configs/macbook.yaml (drive_root -> ~/ckd-edge/mirror)
INFO  Verifying face_path resolution on test CSVs...
INFO    [OK  ] gen1_test.csv : 100/100 sampled face_paths exist
INFO    [OK  ] gen2_test.csv : 100/100 sampled face_paths exist
INFO    [OK  ] gen3_test.csv : 100/100 sampled face_paths exist
INFO  SETUP COMPLETE
```

**If verification shows < 100% existence,** there are likely missing
techniques in the ZIPs you downloaded. Re-check Step 1.3.

---

## 4. Run the Evaluation

### Option A: CLI (recommended for one-shot)

```bash
python scripts/edge/run_edge_eval_macbook.py \
    --config configs/macbook.yaml \
    --output-dir ~/ckd-edge/edge_results \
    --subset-rows 5000 \
    --num-latency-runs 200 \
    --num-warmup 20
```

This runs end-to-end:
1. Pilot conversion + sanity check (`gen3_replay+ewc_seed0`)
2. Mass conversion (remaining 8 checkpoints)
3. AUC eval (all 9 ckpt × up to 4 formats × 3 test splits)
4. Latency benchmark (pilot only, 4 formats × multiple compute units)
5. JSON + Markdown summary in `~/ckd-edge/edge_results/results/`

### Option B: Notebook (recommended for debugging)

```bash
jupyter lab notebooks/08_edge_evaluation_macbook.ipynb
```

Cells 1–8 are the same workflow split into runnable chunks. Use this
when you want to inspect intermediate outputs (sanity check, conversion
artifacts, raw eval results) before continuing.

### What to expect

| Stage | Wall time on M2 Max |
|---|---|
| ZIP extract (Step 3) | ~15 min |
| Pilot conversion + sanity check | ~5 min |
| Mass conversion (8 more ckpts) | ~30–50 min |
| AUC eval (9 ckpt × 4 fmt × 3 split) | ~60–90 min |
| Latency benchmark | ~5 min |
| **Total** | **~2.5–3 hours** |

---

## 5. Interpret the Results

After the run completes, look at:

```
~/ckd-edge/edge_results/
├── conversions/                  <- intermediate ONNX + TFLite + CoreML artifacts
│   ├── gen1_seed0/
│   ├── gen3_replay+ewc_seed0/
│   └── ...
└── results/
    ├── edge_eval_full.json       <- full structured results
    ├── edge_eval_summary.md      <- thesis-ready Markdown
    ├── sanity_report.json        <- numerical sanity vs PyTorch
    └── latency_vs_size.png       <- (if Cell 7 of notebook was run)
```

### Key tables in `edge_eval_summary.md`

1. **Sanity check** — confirms conversion did not corrupt numerics.
   All rows should show `OK` (max_abs_diff < threshold). If `FAIL`,
   see troubleshooting below.

2. **Latency** — mean/p50/p95/p99 per (runtime, mode, compute_unit).
   Expected ranges on M2 Max:
   - TFLite fp32: ~2–5 ms
   - TFLite int8: ~1–3 ms
   - CoreML fp32 (ANE+GPU): ~1–3 ms
   - CoreML int8 (ANE+GPU): ~1–2 ms

3. **AUC mean ± std** by (ckpt_gen, runtime, mode, test_split).
   - PyTorch fp32 = reference
   - TFLite fp32 should match PyTorch within 0.001 AUC
   - TFLite int8 / CoreML int8: should be within ~0.01 AUC of fp32
   - If int8 drops more than 0.02 AUC, your calibration data may
     not be representative — re-check the calibration mix.

### What to copy into the thesis

For Section X "Edge Deployment Feasibility":

- The latency table (4 rows: TFLite fp32/int8, CoreML fp32/int8)
- Model size column from the latency table
- AUC mean ± std for `gen3` row × `gen3_test` (headline number)
- AUC drop fp32 → int8 (quantization cost)

For the appendix:

- Full per-checkpoint AUC table (Section 4 of the Markdown)
- Sanity check table (proof of numerical fidelity)

### Honest hardware framing

In the thesis, frame the M2 Max numbers as:

> "Edge deployment feasibility was evaluated by converting the student
> model to TFLite (fp32 and int8 with mixed-generation calibration) and
> CoreML (fp32 and int8) formats, and benchmarked on Apple M2 Max as a
> representative ARMv8.2+ architecture. M2 Max is a development-class
> chip; smartphone-grade ARM Cortex-A78 or Snapdragon 7-series CPUs are
> expected to be 4–8× slower based on published MobileNetV3-Large
> benchmarks. Evaluation on smartphone- or embedded-class devices is
> identified as future work."

---

## 6. Troubleshooting

### Sanity check fails for int8 (max_abs_diff > 2e-1)

**Likely cause:** insufficient calibration data, or hard-swish/squeeze-excite
quantization drift.

**Fixes (try in order):**
1. Increase `--calibration-batches 32` (default is 16)
2. Verify the calibration CSV in `~/ckd-edge/edge_results/subsets/calibration_mixed.csv`
   has rows from all 3 generations
3. As a last resort, fall back to `--no-coreml` and rely on TFLite only

### CoreML conversion error: `coremltools.optimize.coreml not available`

Your coremltools version is < 7.0. Upgrade:

```bash
uv pip install --upgrade 'coremltools>=7.0'
# or:
pip install --upgrade 'coremltools>=7.0'
```

### `onnx2tf` error during int8 conversion: "Per-channel not supported"

The script auto-retries with `per-tensor` mode. If you see this in the
log, no action needed.

### `tensorflow-macos` install fails

For Python 3.12+, `tensorflow-macos` does not yet have wheels. Either:
- Downgrade to Python 3.11 (recommended)
- Use full `tensorflow` package: `pip install 'tensorflow>=2.13,<2.16'`

### `cv2.imread` returns None

Your face_path in the CSV is wrong. Re-run setup with `--resume` and
check the verification step's output ratio. If still failing, manually
inspect a single row:

```python
import pandas as pd
df = pd.read_csv('~/ckd-edge/mirror/datasets/splits/gen3_test.csv', nrows=5)
print(df['face_path'].tolist())
```

The paths should point to files under `~/ckd-edge/df40/...`.

### MacBook overheating during latency benchmark

Add `--num-latency-runs 100 --num-warmup 10` to reduce thermal stress.
For thesis-grade numbers, ensure the chassis is not throttled — let
the laptop cool between cells if running interactively.

---

## 7. Where to Go Next

- **For thesis figures:** open `latency_vs_size.png` in `edge_results/results/`.
  Consider re-running with `matplotlib` styled to match your thesis.
- **For deeper hardware analysis:** see `scripts/edge/run_edge_eval_macbook.py`
  `--num-latency-runs 1000` for tighter p99 estimates.
- **For cross-checking int8 calibration quality:** inspect
  `~/ckd-edge/edge_results/sanity_report.json` for the cosine similarity
  numbers — values > 0.999 indicate near-perfect fidelity.
- **For deployment beyond MacBook:** the `.tflite` files in
  `edge_results/conversions/<run_tag>/tflite/` are deployable to Android
  via TFLite Java API or to iOS via the same. The `.mlpackage` bundles
  are deployable to iOS/macOS via CoreML.
