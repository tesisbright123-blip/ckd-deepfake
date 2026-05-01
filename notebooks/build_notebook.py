"""One-shot generator for notebooks/colab_run_all.ipynb.

Run from repo root:   python notebooks/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


CELLS: list[dict] = []

CELLS.append(md(
    "# CKD-Deepfake — End-to-End Colab Pipeline\n"
    "\n"
    "Thesis: **Continual Knowledge Distillation for Cross-Generational Deepfake Detection on Edge Devices**.\n"
    "\n"
    "This notebook runs the full pipeline on Google Colab with a single A100/T4 GPU:\n"
    "\n"
    "1. Mount Drive + clone private repo\n"
    "2. Download DF40 (fake crops) + DF40 real crops + teacher weights\n"
    "3. Catalog DF40 into `metadata_<gen>.csv` (gen1 = classic face-swap, gen2 = reenactment, gen3 = diffusion/modern)\n"
    "4. Generate splits + ensemble soft labels\n"
    "5. Initial distillation on gen1, then continual distillation on gen2 → gen3\n"
    "6. Edge evaluation (TFLite fp32/fp16/int8) + generate figures\n"
    "\n"
    "**Dataset choice.** All three generations come from the **DF40 dataset** — it bundles 40 techniques covering "
    "classic face-swap → reenactment → modern diffusion. One dataset, consistent face-crop format (224×224), "
    "no per-video face extraction required.\n"
))

CELLS.append(md("## 0. Sanity check: GPU + Python"))
CELLS.append(code(
    "!nvidia-smi || echo 'No GPU — switch runtime to GPU before running.'\n"
    "import sys, platform\n"
    "print('Python:', sys.version)\n"
    "print('Platform:', platform.platform())\n"
))

CELLS.append(md("## 1. Mount Google Drive"))
CELLS.append(code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "\n"
    "from pathlib import Path\n"
    "DRIVE_ROOT = Path('/content/drive/MyDrive/CKD_Thesis')\n"
    "DRIVE_ROOT.mkdir(parents=True, exist_ok=True)\n"
    "for sub in ['datasets/raw', 'datasets/faces', 'datasets/splits',\n"
    "            'checkpoints/teachers', 'checkpoints/students',\n"
    "            'soft_labels', 'runs', 'results', 'results/figures']:\n"
    "    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)\n"
    "print('Drive root:', DRIVE_ROOT)\n"
))

CELLS.append(md(
    "## 2. Clone the repository\n"
    "\n"
    "The repo is private. You'll need a [GitHub personal access token](https://github.com/settings/tokens) "
    "with `repo` scope. It's only used to clone and is not stored.\n"
))
CELLS.append(code(
    "import getpass, os, subprocess\n"
    "from pathlib import Path\n"
    "\n"
    "REPO_USER = 'tesisbright123-blip'\n"
    "REPO_NAME = 'ckd-deepfake'\n"
    "REPO_DIR  = Path('/content') / REPO_NAME\n"
    "\n"
    "if not REPO_DIR.exists():\n"
    "    token = getpass.getpass('GitHub token (repo scope): ').strip()\n"
    "    url = f'https://{REPO_USER}:{token}@github.com/{REPO_USER}/{REPO_NAME}.git'\n"
    "    subprocess.run(['git', 'clone', '--depth', '1', url, str(REPO_DIR)], check=True)\n"
    "    del token\n"
    "else:\n"
    "    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=False)\n"
    "\n"
    "os.chdir(REPO_DIR)\n"
    "print('Working dir:', os.getcwd())\n"
    "!ls\n"
))

CELLS.append(md("## 3. Install dependencies"))
CELLS.append(code(
    "!pip install -q -r requirements.txt\n"
    "!pip install -q gdown tqdm\n"
))

CELLS.append(md(
    "## 4. Download + extract DF40 to Drive (one-time, ~55 GB compressed → ~70 GB extracted)\n"
    "\n"
    "All download + extract logic lives in `scripts/00_data_prep.py` so it survives:\n"
    "- runtime disconnects (re-runs pick up exactly where they stopped, via `.extracted_ok` markers)\n"
    "- gdrive quota errors (per-file failures are logged, batch continues)\n"
    "- partial extracts (folder + leftover ZIP → auto-wipe + re-extract)\n"
    "- laptop crashes (no in-cell state to lose; everything is a script + Drive)\n"
    "\n"
    "Three actions:\n"
    "- `--action status` — print what's done / partial / missing (no side effects)\n"
    "- `--action download` — resume gdown of DF40_train folder + 2 real ZIPs\n"
    "- `--action extract` — extract every ZIP that hasn't been marked extracted\n"
    "- `--action all` — status → download → extract → status\n"
    "\n"
    "Re-run this cell as many times as needed. It's safe and idempotent.\n"
))
CELLS.append(code(
    "# Quick status check first (no side effects).\n"
    "!python scripts/00_data_prep.py --action status\n"
))
CELLS.append(code(
    "# Full pipeline: download anything missing, then extract everything.\n"
    "# Pixart alone takes ~3-5h to extract on Drive (13.8GB ZIP).\n"
    "# Total wall time ~3-6 h for a fresh run (less if resuming).\n"
    "!python scripts/00_data_prep.py --action all\n"
))

CELLS.append(md(
    "## 5. Download teacher checkpoints (auto)\n"
    "\n"
    "Three teachers, all downloaded automatically and cached on Drive:\n"
    "\n"
    "| Model | Source | Size |\n"
    "|---|---|---|\n"
    "| EfficientNet-B4 | DeepfakeBench GitHub release | ~68 MB |\n"
    "| Recce | DeepfakeBench GitHub release | ~183 MB |\n"
    "| CLIP ViT-L/14 (CLIPping head) | Google Drive | ~1.7 GB |\n"
    "\n"
    "Cell is idempotent — skips files that already exist.\n"
))
CELLS.append(code(
    "import os, subprocess\n"
    "from pathlib import Path\n"
    "\n"
    "teachers_dir = Path('/content/drive/MyDrive/CKD_Thesis/checkpoints/teachers')\n"
    "teachers_dir.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "DOWNLOADS = [\n"
    "    {\n"
    "        'name': 'efficientnet_b4.pth',\n"
    "        'kind': 'wget',\n"
    "        'url':  'https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/effnb4_best.pth',\n"
    "    },\n"
    "    {\n"
    "        'name': 'recce.pth',\n"
    "        'kind': 'wget',\n"
    "        'url':  'https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/recce_best.pth',\n"
    "    },\n"
    "    {\n"
    "        'name': 'clip_clipping.pth',\n"
    "        'kind': 'gdown',\n"
    "        'id':   '1jtkBoIuMw5wrooTv-anh668G3_Xt-UnX',\n"
    "    },\n"
    "]\n"
    "\n"
    "for item in DOWNLOADS:\n"
    "    dst = teachers_dir / item['name']\n"
    "    if dst.is_file() and dst.stat().st_size > 1_000_000:\n"
    "        print(f\"[skip] {item['name']} already present ({dst.stat().st_size/1e6:.1f} MB)\")\n"
    "        continue\n"
    "    print(f\"[download] {item['name']} ...\")\n"
    "    if item['kind'] == 'wget':\n"
    "        subprocess.run(\n"
    "            ['wget', '-q', '--show-progress', item['url'], '-O', str(dst)],\n"
    "            check=True,\n"
    "        )\n"
    "    else:  # gdown\n"
    "        import gdown\n"
    "        gdown.download(id=item['id'], output=str(dst), quiet=False, fuzzy=True)\n"
    "    print(f\"  saved -> {dst} ({dst.stat().st_size/1e6:.1f} MB)\")\n"
    "\n"
    "print('\\n=== Teacher weights ===')\n"
    "for fname in ['efficientnet_b4.pth', 'recce.pth', 'clip_clipping.pth']:\n"
    "    p = teachers_dir / fname\n"
    "    status = f'OK  ({p.stat().st_size/1e6:.1f} MB)' if p.is_file() else 'MISSING'\n"
    "    print(f'  {status:20s}  {fname}')\n"
))

CELLS.append(md(
    "## 6. Catalog DF40 → `metadata_<gen>.csv`\n"
    "\n"
    "Technique lists per generation come from `configs/default.yaml`. The catalog script:\n"
    "\n"
    "- Walks every requested technique under `datasets/raw/df40/`, handles `ff/`, `cdf/`, `fake/` and "
    "  flat layouts.\n"
    "- Pulls real frames from `datasets/raw/df40_real/` and labels them 0.\n"
    "- Writes the canonical 7-column metadata CSV consumed by step 02.\n"
))
CELLS.append(code(
    "for gen in ['gen1', 'gen2', 'gen3']:\n"
    "    print(f'=== Cataloging {gen} ===')\n"
    "    !python scripts/01b_catalog_df40.py --config configs/default.yaml --generation $gen\n"
    "\n"
    "import pandas as pd\n"
    "for gen in ['gen1', 'gen2', 'gen3']:\n"
    "    p = f'/content/drive/MyDrive/CKD_Thesis/datasets/faces/metadata_{gen}.csv'\n"
    "    df = pd.read_csv(p)\n"
    "    print(f'{gen}: rows={len(df)} real={(df.label==0).sum()} fake={(df.label==1).sum()} '\n"
    "          f'techniques={df.technique.nunique()}')\n"
))

CELLS.append(md("## 7. Generate 70/15/15 video-level splits"))
CELLS.append(code(
    "for gen in ['gen1', 'gen2', 'gen3']:\n"
    "    !python scripts/02_generate_splits.py --config configs/default.yaml --generation $gen --seed 0\n"
    "\n"
    "!ls /content/drive/MyDrive/CKD_Thesis/datasets/splits/\n"
))

CELLS.append(md(
    "## 8. Generate ensemble soft labels (teacher predictions)\n"
    "\n"
    "Runs all three teachers over train/val frames per generation and caches the softmax average. "
    "Slowest step in the pipeline — budget ~1 h per generation on A100.\n"
))
CELLS.append(code(
    "for gen in ['gen1', 'gen2', 'gen3']:\n"
    "    print(f'=== Soft labels for {gen} ===')\n"
    "    !python scripts/03_generate_soft_labels.py --config configs/default.yaml --generation $gen\n"
))

CELLS.append(md(
    "## 9. (Optional) Mirror hot data to Colab local SSD\n"
    "\n"
    "Reading 100 k face crops from Drive is slow. If runtime disk has enough room, rsync the current "
    "generation's frames + soft labels to `/content/ckd_local` before training.\n"
))
CELLS.append(code(
    "import shutil\n"
    "src = '/content/drive/MyDrive/CKD_Thesis'\n"
    "dst = '/content/ckd_local'\n"
    "# Flip to True to enable.\n"
    "ENABLE_HOTDATA = False\n"
    "if ENABLE_HOTDATA:\n"
    "    !mkdir -p $dst/datasets $dst/soft_labels\n"
    "    !rsync -a --info=progress2 $src/datasets/splits/ $dst/datasets/splits/\n"
    "    !rsync -a --info=progress2 $src/soft_labels/ $dst/soft_labels/\n"
    "    print('Done mirroring. Point --hotdata-root at', dst, 'if your training script supports it.')\n"
    "else:\n"
    "    print('Hotdata mirroring disabled.')\n"
))

CELLS.append(md("## 10. Initial distillation on gen1 (student = MobileNetV3-Large)"))
CELLS.append(code(
    "!python scripts/04_initial_distillation.py --config configs/default.yaml --generation gen1\n"
))

CELLS.append(md(
    "## 11. Continual distillation: gen1 → gen2\n"
    "\n"
    "Anti-forgetting method is selected via `--method` (one of `replay`, `ewc`, `lwf`). "
    "We default to **replay** (small rehearsal buffer — typically best AUC/forgetting trade-off).\n"
))
CELLS.append(code(
    "PREV = '/content/drive/MyDrive/CKD_Thesis/checkpoints/students/gen1/best.pth'\n"
    "!python scripts/05_continual_distillation.py --config configs/default.yaml \\\n"
    "    --generation gen2 --method replay --previous-checkpoint $PREV\n"
))

CELLS.append(md(
    "## 12. Continual distillation: gen2 → gen3\n"
    "\n"
    "Previous checkpoint path suffix changes for gen2 because script 05 writes under "
    "`checkpoints/students/{gen}_{method}/best.pth` (so gen2+replay lives at `gen2_replay/`).\n"
))
CELLS.append(code(
    "PREV = '/content/drive/MyDrive/CKD_Thesis/checkpoints/students/gen2_replay/best.pth'\n"
    "!python scripts/05_continual_distillation.py --config configs/default.yaml \\\n"
    "    --generation gen3 --method replay --previous-checkpoint $PREV\n"
))

CELLS.append(md(
    "## 13. Edge evaluation (TFLite fp32 / fp16 / int8)\n"
    "\n"
    "Converts each generation's best checkpoint to TFLite variants and benchmarks inference "
    "latency + accuracy on the test split.\n"
))
CELLS.append(code(
    "# gen1 comes from scripts/04 (no --method). gen2/gen3 come from\n"
    "# scripts/05 with method=replay, so we pass --method replay to help\n"
    "# the script resolve the right checkpoint subdir.\n"
    "for gen in ['gen1', 'gen2', 'gen3']:\n"
    "    method_flag = '' if gen == 'gen1' else '--method replay'\n"
    "    !python scripts/07_edge_evaluation.py --config configs/default.yaml \\\n"
    "        --generation $gen $method_flag --modes fp32,fp16,int8 --num-runs 100\n"
))

CELLS.append(md("## 14. Generate thesis figures"))
CELLS.append(code(
    "!python scripts/08_generate_figures.py --config configs/default.yaml\n"
    "!ls /content/drive/MyDrive/CKD_Thesis/results/figures\n"
))

CELLS.append(md(
    "---\n"
    "## Done 🎉\n"
    "\n"
    "Deliverables on Drive:\n"
    "\n"
    "- `checkpoints/students/gen{1,2,3}/best.pth` — student after each stage\n"
    "- `results/edge_*.json` — TFLite latency/accuracy\n"
    "- `results/figures/*.pdf` — thesis-ready plots (CDE, CGRS, S1–S3 heatmaps, latency bars)\n"
))


def main() -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
        },
        "cells": CELLS,
    }
    out = Path(__file__).resolve().parent / "colab_run_all.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
