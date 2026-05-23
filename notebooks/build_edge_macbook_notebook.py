"""Generator for notebooks/08_edge_evaluation_macbook.ipynb.

Run from repo root:
    python notebooks/build_edge_macbook_notebook.py

This rebuilds the comprehensive MacBook edge-eval notebook end-to-end.
The output notebook is designed to be opened in VSCode/Jupyter on a
MacBook Apple Silicon machine — every cell has:
  - A markdown header explaining purpose
  - Expected output description
  - Error handling / fallbacks
  - Resume-friendly side effects (per-ckpt JSON, drive sync, etc.)

The notebook does NOT require running setup scripts manually — it
calls them internally via subprocess so the user can do everything
from inside VSCode.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


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


# =============================================================================
# Title + Overview
# =============================================================================
CELLS.append(md(dedent("""
    # CKD Edge Evaluation — MacBook M2 Max (Full End-to-End)

    **Thesis:** Continual Knowledge Distillation for Cross-Generational
    Deepfake Detection on Edge Devices.

    This notebook runs the complete edge-deployment evaluation pipeline on a
    MacBook Apple Silicon (M-series) machine. Use the 9 student checkpoints
    trained earlier in Colab (3 generations × 3 seeds), convert each to
    TFLite (fp32+int8) and CoreML (fp32+int8), measure AUC + latency, and
    output a thesis-ready summary.

    ## How to use this notebook

    1. Open this `.ipynb` in **VSCode** (recommended) or Jupyter on your MacBook.
    2. Set the paths in **Cell 3** to point at the files you downloaded from
       Google Drive.
    3. Run cells **top to bottom**. Each cell starts with a markdown header
       explaining its purpose and expected output.
    4. **Cell 6 (pilot conversion)** is the safety gate — stop and fix if its
       sanity check fails before continuing.
    5. **Cell 8 (mass eval)** is the long one (~2–3 hours). It uses per-checkpoint
       caching, so if your MacBook crashes, just re-run the same cell — `--resume`
       skips work that's already done.
    6. **Cell 12** generates the markdown summary you'll paste into the thesis.

    ## Prerequisites checklist

    - [ ] MacBook with Apple Silicon (M1/M2/M3/M4) — Intel MacBooks won't run CoreML on ANE
    - [ ] macOS 13 (Ventura) or newer
    - [ ] At least **120 GB free disk space** (76 GB ZIPs + ~80 GB extracted + 5 GB env)
    - [ ] Repo cloned: `git clone https://github.com/tesisbright123-blip/ckd-deepfake.git`
    - [ ] Drive ZIPs downloaded to a folder you control (see Cell 3 layout)

    ## What this notebook does NOT do

    - **No training.** All checkpoints come from previous Colab runs.
    - **No Drive integration.** Files must be locally downloaded first.
    - **No internet during runtime** except for the one-time `pip install`.
""").lstrip()))


# =============================================================================
# CELL 1 — Hardware + macOS version check
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 1 — Hardware and OS sanity check

    **Purpose:** Verify this notebook is running on a supported platform before
    we burn time installing dependencies.

    **Expected output:**
    ```
    Python   : 3.11.x (or 3.10.x — both supported)
    Platform : macOS-14.x-arm64-arm-64bit (Sonoma) or similar
    Machine  : arm64
    CPU cores: 12 (M2 Max) or similar
    RAM      : 32.0 GB total, X GB available
    ```

    **If you see `Machine: x86_64`** → you're on an Intel Mac. The notebook will
    still produce TFLite results but CoreML benchmarks won't show ANE numbers.

    **If `Platform` is not Darwin** → wrong machine. Stop here.
""").lstrip()))

CELLS.append(code(dedent("""
    import platform, sys, os

    print('Python   :', sys.version.split()[0])
    print('Platform :', platform.platform())
    print('Machine  :', platform.machine())
    print('CPU cores:', os.cpu_count())

    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f'RAM      : {mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB available')
    except ImportError:
        print('RAM      : (psutil not installed yet)')

    # Hard checks
    is_macos = platform.system() == 'Darwin'
    is_arm64 = platform.machine() == 'arm64'

    if not is_macos:
        raise RuntimeError(
            f'This notebook expects macOS. Detected {platform.system()!r}. Stop and re-run on your MacBook.'
        )
    if not is_arm64:
        print('\\n⚠ Intel Mac detected — CoreML ANE benchmarks will be unavailable. Other parts will work.')
    else:
        print('\\n✅ Apple Silicon confirmed.')
""").lstrip()))


# =============================================================================
# CELL 2 — Locate repo + setup paths
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 2 — Locate the repo

    **Purpose:** Find the cloned `ckd-deepfake` repo root, regardless of where
    you opened this notebook from. The repo path is needed for relative imports
    and for invoking the setup/eval scripts.

    **Expected output:**
    ```
    Notebook is in : /Users/<you>/ckd-deepfake/notebooks
    REPO_ROOT      : /Users/<you>/ckd-deepfake
    scripts/edge/  : exists ✅
    configs/       : exists ✅
    ```

    **If `scripts/edge/ not found`** → you opened this from outside the repo.
    Re-clone the repo and open the notebook from `<repo>/notebooks/`.
""").lstrip()))

CELLS.append(code(dedent("""
    from pathlib import Path
    import sys

    # Walk upward from cwd to find the repo root (contains scripts/edge).
    cwd = Path.cwd().resolve()
    REPO_ROOT = cwd
    for _ in range(6):
        if (REPO_ROOT / 'scripts' / 'edge').is_dir() and (REPO_ROOT / 'configs').is_dir():
            break
        REPO_ROOT = REPO_ROOT.parent
    else:
        raise RuntimeError(
            'Could not locate the ckd-deepfake repo root by walking up from cwd. '
            f'Started at {cwd}. Re-clone the repo and open this notebook from <repo>/notebooks/.'
        )

    print(f'Notebook is in : {cwd}')
    print(f'REPO_ROOT      : {REPO_ROOT}')
    print(f'scripts/edge/  : exists {\"✅\" if (REPO_ROOT / \"scripts/edge\").is_dir() else \"❌\"}')
    print(f'configs/       : exists {\"✅\" if (REPO_ROOT / \"configs\").is_dir() else \"❌\"}')

    # Make repo importable
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
""").lstrip()))


# =============================================================================
# CELL 3 — User-configurable paths
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 3 — Configure your download paths

    **Purpose:** Tell the notebook where you saved the files downloaded from
    Google Drive. These three paths are the ONLY user-configurable bits — change
    them to match where you put your files.

    **Required folder layout:**

    ```
    DOWNLOAD_DIR/
    ├── df40_zips/                    ← 31 ZIPs (see list below)
    │   ├── blendface.zip
    │   ├── ...
    │   ├── FaceForensics++_real_data_for_DF40.zip
    │   └── Celeb-DF-v2_real_data_for_DF40.zip
    ├── ckd_ckpts/                    ← 9 checkpoint subfolders
    │   ├── gen1_seed0/best.pth
    │   ├── ...
    │   └── gen3_replay+ewc_seed2/best.pth
    └── ckd_splits/                   ← 3 test CSVs
        ├── gen1_test.csv
        ├── gen2_test.csv
        └── gen3_test.csv
    ```

    **Required ZIPs (31 total):**

    | Group | ZIPs |
    |---|---|
    | **Gen1** (9) | blendface, e4s, facedancer, faceswap, fsgan, inswap, mobileswap, simswap, uniface |
    | **Gen2** (11) | fomm, facevid2vid, wav2lip, MRAA, one_shot_free, pirender, tpsm, lia, danet, sadtalker, mcnet |
    | **Gen3** (9) | sd2.1, ddim, pixart, DiT, SiT, StyleGAN2, StyleGAN3, StyleGANXL, VQGAN |
    | **Real** (2) | FaceForensics++_real_data_for_DF40, Celeb-DF-v2_real_data_for_DF40 |

    **NOT needed:** `hyperreenact.zip` — it's in the Drive backup but never made
    it into our train/test splits (configs/default.yaml has it commented out).

    **Expected output:** A green checklist showing all 31 ZIPs + 9 checkpoint
    folders + 3 CSVs present. If any are missing, the cell prints the list and
    raises an error.
""").lstrip()))

CELLS.append(code(dedent("""
    from pathlib import Path

    # ⚠⚠⚠ EDIT THESE PATHS to match where you saved your downloads ⚠⚠⚠
    DOWNLOAD_DIR = Path.home() / 'Downloads' / 'ckd-edge-prep'

    ZIP_DIR  = DOWNLOAD_DIR / 'df40_zips'
    CKPT_DIR = DOWNLOAD_DIR / 'ckd_ckpts'
    CSV_DIR  = DOWNLOAD_DIR / 'ckd_splits'

    # Where the extracted local mirror will live (script will create this).
    OUTPUT_ROOT = Path.home() / 'ckd-edge'

    # Where edge results land (per-ckpt JSON, summary, figures).
    EDGE_RESULTS = OUTPUT_ROOT / 'edge_results'

    print(f'DOWNLOAD_DIR : {DOWNLOAD_DIR}')
    print(f'ZIP_DIR      : {ZIP_DIR}')
    print(f'CKPT_DIR     : {CKPT_DIR}')
    print(f'CSV_DIR      : {CSV_DIR}')
    print(f'OUTPUT_ROOT  : {OUTPUT_ROOT}')
    print(f'EDGE_RESULTS : {EDGE_RESULTS}')
    print()

    # --- Expected files ---
    REQUIRED_ZIPS = [
        # gen1 (9)
        'blendface.zip', 'e4s.zip', 'facedancer.zip', 'faceswap.zip', 'fsgan.zip',
        'inswap.zip', 'mobileswap.zip', 'simswap.zip', 'uniface.zip',
        # gen2 (11)
        'fomm.zip', 'facevid2vid.zip', 'wav2lip.zip', 'MRAA.zip', 'one_shot_free.zip',
        'pirender.zip', 'tpsm.zip', 'lia.zip', 'danet.zip', 'sadtalker.zip', 'mcnet.zip',
        # gen3 (9)
        'sd2.1.zip', 'ddim.zip', 'pixart.zip', 'DiT.zip', 'SiT.zip',
        'StyleGAN2.zip', 'StyleGAN3.zip', 'StyleGANXL.zip', 'VQGAN.zip',
        # real (2)
        'FaceForensics++_real_data_for_DF40.zip', 'Celeb-DF-v2_real_data_for_DF40.zip',
    ]
    REQUIRED_CKPTS = [
        'gen1_seed0/best.pth', 'gen1_seed1/best.pth', 'gen1_seed2/best.pth',
        'gen2_replay+ewc_seed0/best.pth', 'gen2_replay+ewc_seed1/best.pth',
        'gen2_replay+ewc_seed2/best.pth',
        'gen3_replay+ewc_seed0/best.pth', 'gen3_replay+ewc_seed1/best.pth',
        'gen3_replay+ewc_seed2/best.pth',
    ]
    REQUIRED_CSVS = ['gen1_test.csv', 'gen2_test.csv', 'gen3_test.csv']

    # --- Verify ---
    def _check_files(label, base_dir, files):
        missing = []
        total_size = 0
        for f in files:
            p = base_dir / f
            if p.is_file():
                total_size += p.stat().st_size
            else:
                missing.append(f)
        present = len(files) - len(missing)
        status = '✅' if not missing else '⚠'
        print(f'{status} {label}: {present}/{len(files)} present ({total_size / 1e9:.1f} GB)')
        return missing

    missing_zips  = _check_files('ZIPs       ', ZIP_DIR,  REQUIRED_ZIPS)
    missing_ckpts = _check_files('Checkpoints', CKPT_DIR, REQUIRED_CKPTS)
    missing_csvs  = _check_files('CSVs       ', CSV_DIR,  REQUIRED_CSVS)

    if missing_zips:
        print('\\nMissing ZIPs:')
        for f in missing_zips:
            print(f'   - {f}')
    if missing_ckpts:
        print('\\nMissing checkpoints:')
        for f in missing_ckpts:
            print(f'   - {f}')
    if missing_csvs:
        print('\\nMissing CSVs:')
        for f in missing_csvs:
            print(f'   - {f}')

    if missing_zips or missing_ckpts or missing_csvs:
        raise RuntimeError(
            'Some required files are missing. Download them from Google Drive '
            '(see Cell 3 docstring) and re-run this cell.'
        )

    # --- Check disk space ---
    import shutil
    total, used, free = shutil.disk_usage(Path.home())
    print(f'\\nDisk free at {Path.home()}: {free / 1e9:.1f} GB / {total / 1e9:.1f} GB total')
    if free < 120e9:
        print(f'⚠ Less than 120 GB free. Extraction + intermediate files need ~100 GB extra.')
        print('   Consider freeing space before proceeding to Cell 5.')
    else:
        print('✅ Enough disk space.')
""").lstrip()))


# =============================================================================
# CELL 4 — Install / verify dependencies
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 4 — Install dependencies (one-time per env)

    **Purpose:** Install Python 3.11 venv with all required packages.

    **What this cell does:**
    1. Checks if a venv at `<repo>/.venv` already exists with Python 3.11.
    2. If not, installs `uv` (a fast pip alternative), creates a Python 3.11 venv,
       and installs `requirements-macbook.txt`.
    3. Verifies critical imports succeed.

    **Expected output:**
    ```
    ✅ uv already installed
    ✅ .venv exists at /Users/<you>/ckd-deepfake/.venv (Python 3.11.x)
    ✅ torch        : 2.x.x
    ✅ onnx         : 1.x.x
    ✅ onnx2tf      : 1.22.x
    ✅ coremltools  : 7.x.x
    ✅ tensorflow   : 2.15.x
    ```

    **Important:** This cell runs commands inside a subprocess, but doesn't
    automatically activate the venv for THIS notebook kernel. After this cell
    completes, you need to:
    1. Stop the kernel
    2. Re-open this notebook with the `.venv` Python interpreter selected
       (in VSCode: bottom-right interpreter picker → choose `<repo>/.venv/bin/python`)

    Once you re-launch with the right kernel, re-run cells 1-3, then skip this cell
    and continue from Cell 5.

    **Time:** ~5 minutes first run, instant on subsequent runs.
""").lstrip()))

CELLS.append(code(dedent("""
    import subprocess, shutil, sys, os
    from pathlib import Path

    VENV_DIR = REPO_ROOT / '.venv'
    PYTHON_VERSION = '3.11'

    def _run(cmd, **kwargs):
        print(f'$ {\" \".join(str(c) for c in cmd)}')
        return subprocess.run(cmd, check=True, **kwargs)

    # --- 1. Install uv if missing ---
    if shutil.which('uv') is None:
        print('Installing uv (fast pip alternative)...')
        _run(['sh', '-c', 'curl -LsSf https://astral.sh/uv/install.sh | sh'])
        # uv installs to ~/.local/bin; add to PATH for this Python process
        local_bin = str(Path.home() / '.local' / 'bin')
        if local_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = f'{local_bin}:{os.environ.get(\"PATH\", \"\")}'
    else:
        print(f'✅ uv already installed: {shutil.which(\"uv\")}')

    # --- 2. Create venv if missing ---
    venv_python = VENV_DIR / 'bin' / 'python'
    if not venv_python.is_file():
        print(f'\\nCreating venv at {VENV_DIR} with Python {PYTHON_VERSION}...')
        _run(['uv', 'venv', str(VENV_DIR), '--python', PYTHON_VERSION])
    else:
        # Check the venv's Python version matches.
        ver_out = subprocess.check_output([str(venv_python), '--version'], text=True).strip()
        print(f'✅ Venv exists: {ver_out} at {VENV_DIR}')

    # --- 3. Install requirements ---
    req_file = REPO_ROOT / 'requirements-macbook.txt'
    if not req_file.is_file():
        raise FileNotFoundError(f'{req_file} not in repo — pull the latest main.')

    print('\\nInstalling requirements (this is a no-op if already up to date)...')
    _run(['uv', 'pip', 'install', '--python', str(venv_python),
          '-r', str(req_file)])
    _run(['uv', 'pip', 'install', '--python', str(venv_python),
          '-e', str(REPO_ROOT)])

    # --- 4. Verify imports IN THE VENV ---
    verify_script = (
        'import torch, onnx, onnx2tf, coremltools, tensorflow as tf, platform\\n'
        'print(f\"torch        : {torch.__version__}\")\\n'
        'print(f\"onnx         : {onnx.__version__}\")\\n'
        'print(f\"onnx2tf      : {getattr(onnx2tf, \\\"__version__\\\", \\\"unknown\\\")}\")\\n'
        'print(f\"coremltools  : {coremltools.__version__}\")\\n'
        'print(f\"tensorflow   : {tf.__version__}\")\\n'
        'print(f\"macOS arm64  : {platform.system() == \\\"Darwin\\\" and platform.machine() == \\\"arm64\\\"}\")\\n'
    )
    print('\\n--- Verifying imports inside venv ---')
    result = subprocess.run(
        [str(venv_python), '-c', verify_script],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print('STDERR:')
        print(result.stderr)
        raise RuntimeError('Dependency verification failed — see stderr above.')

    print(f'\\n✅ Dependencies installed in {VENV_DIR}')
    print(f'\\n👉 NEXT STEP: switch this notebook\\'s kernel to {venv_python}')
    print('   In VSCode: click the interpreter picker at the top-right and pick the .venv Python.')
    print('   Then re-run Cells 1-3 with the new kernel, and continue from Cell 5.')
""").lstrip()))


# =============================================================================
# CELL 5 — Confirm we're running with the .venv Python
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 5 — Confirm the venv Python is active

    **Purpose:** This is a guard cell. If you didn't switch the kernel after
    Cell 4, the rest of the notebook will fail with ImportError. This cell
    detects that and stops you early.

    **Expected output:**
    ```
    sys.executable : /Users/<you>/ckd-deepfake/.venv/bin/python
    ✅ Running inside the .venv — proceed to Cell 6
    ```

    **If you see something else** (e.g. `/usr/bin/python3` or the system Python),
    stop and switch the kernel:
    - **VSCode:** Click the kernel picker at the top right → "Select Another Kernel" → "Python Environments" → pick `.venv (Python 3.11.x)`
    - **Jupyter Lab:** Kernel menu → Change Kernel → select the venv
""").lstrip()))

CELLS.append(code(dedent("""
    import sys
    from pathlib import Path

    expected = (REPO_ROOT / '.venv' / 'bin' / 'python').resolve()
    actual = Path(sys.executable).resolve()

    print(f'sys.executable : {actual}')
    print(f'expected       : {expected}')

    if actual != expected:
        print('\\n⚠ You are NOT using the .venv Python.')
        print('   Switch the notebook kernel to the .venv Python (see Cell 5 docstring).')
        print('   The rest of this notebook will fail with ImportError otherwise.')
        # Don't hard-raise, allow the user to override if they know what they're doing
        # (e.g. they installed globally and want to use that).
    else:
        print('\\n✅ Running inside the .venv — proceed to Cell 6.')

    # Confirm critical imports work in THIS kernel
    try:
        import torch, onnx, onnx2tf, coremltools
        print(f'\\ntorch       : {torch.__version__}')
        print(f'onnx        : {onnx.__version__}')
        print(f'coremltools : {coremltools.__version__}')
    except ImportError as exc:
        print(f'\\n❌ Import failed in this kernel: {exc}')
        print('   Switch the kernel to the .venv Python and re-run.')
        raise
""").lstrip()))


# =============================================================================
# CELL 6 — Run setup_macbook_mirror.py
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 6 — Extract ZIPs + stage checkpoints + rewrite CSVs

    **Purpose:** This calls `scripts/edge/setup_macbook_mirror.py`, which:
    1. Extracts all 31 ZIPs from `ZIP_DIR` into `OUTPUT_ROOT/df40/` and
       `OUTPUT_ROOT/df40_real/`.
    2. Copies 9 student checkpoints into `OUTPUT_ROOT/mirror/checkpoints/students/`.
    3. Reads each test CSV and rewrites `face_path` from Colab paths to your
       MacBook paths.
    4. Generates `configs/macbook.yaml` with `drive_root` pointing at the mirror.
    5. Verifies file existence on a sample of paths.

    **Resume:** This cell uses `--resume`, so if it crashes mid-extract (e.g.
    you close the lid and macOS sleeps), re-running picks up where it left off.

    **Expected output (last lines):**
    ```
    INFO  Step 4/4: wrote configs/macbook.yaml (drive_root -> /Users/<you>/ckd-edge/mirror)
    INFO  Verifying face_path resolution on test CSVs...
    INFO    [OK  ] gen1_test.csv : 100/100 sampled face_paths exist
    INFO    [OK  ] gen2_test.csv : 100/100 sampled face_paths exist
    INFO    [OK  ] gen3_test.csv : 100/100 sampled face_paths exist
    INFO  SETUP COMPLETE
    ```

    **Time:** ~15-25 minutes (ZIP extraction is the bottleneck on M2 Max).

    **If verification shows < 100% existence** → some ZIP is missing or
    failed to extract. Re-check Cell 3's list and re-run this cell.
""").lstrip()))

CELLS.append(code(dedent("""
    import subprocess, sys

    cmd = [
        sys.executable,
        str(REPO_ROOT / 'scripts/edge/setup_macbook_mirror.py'),
        '--zip-dir',     str(ZIP_DIR),
        '--ckpt-dir',    str(CKPT_DIR),
        '--csv-dir',     str(CSV_DIR),
        '--output-root', str(OUTPUT_ROOT),
        '--resume',
    ]
    print('Running:')
    print('  ' + ' \\\\\\n  '.join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f'Mirror setup failed with exit code {result.returncode}. '
            'Inspect the log above and re-run with --resume to continue.'
        )

    # Sanity check the config got written.
    CONFIG_PATH = REPO_ROOT / 'configs' / 'macbook.yaml'
    if not CONFIG_PATH.is_file():
        raise RuntimeError(f'Expected {CONFIG_PATH} to exist after setup — not found.')

    print(f'\\n✅ Mirror setup complete. Config at: {CONFIG_PATH}')
""").lstrip()))


# =============================================================================
# CELL 7 — Pilot conversion + sanity check
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 7 — Pilot conversion + numerical sanity check

    **Purpose:** Before running the full eval, convert ONE checkpoint
    (`gen3_replay+ewc_seed0`) to all 4 formats (TFLite fp32/int8, CoreML fp32/int8)
    and verify each runtime produces outputs numerically close to PyTorch.

    This is your safety gate — if any sanity check fails, fix it BEFORE the
    long mass-conversion cell. A failed sanity check almost always means
    HardSwish or Squeeze-Excite quantization drift in MobileNetV3-Large.

    **Expected output (key lines):**
    ```
    INFO  Pilot conversion: gen3_replay+ewc_seed0
    INFO  Loaded pilot best.pth (epoch=9, best_val_auc=0.8648)
    INFO  TFLite fp32: ... (X.XX MB)
    INFO  TFLite int8: ... (X.XX MB)
    INFO  CoreML saved: ... fp32.mlpackage (X.XX MB)
    INFO  CoreML saved: ... int8.mlpackage (X.XX MB)
    INFO  Sanity check:
      [PASS]  tflite  fp32  max_abs=X.XXe-XX  cos=0.9999  (threshold 5.00e-04)
      [PASS]  tflite  int8  max_abs=X.XXe-XX  cos=0.9999  (threshold 2.00e-01)
      [PASS]  coreml  fp32  max_abs=X.XXe-XX  cos=0.9999  (threshold 5.00e-04)
      [PASS]  coreml  int8  max_abs=X.XXe-XX  cos=0.9999  (threshold 2.00e-01)
    ```

    **All `[PASS]`** → green light for Cell 8.

    **Any `[FAIL]`** → STOP. Common fixes:
    - If TFLite int8 fails: re-run with `CALIBRATION_BATCHES = 32` (set below)
    - If CoreML int8 fails: check coremltools version is ≥ 7.0
    - If TFLite fp32 fails: likely an opset issue — file an issue with logs

    **Time:** ~5-10 minutes.
""").lstrip()))

CELLS.append(code(dedent("""
    import subprocess, sys

    CALIBRATION_BATCHES = 16  # bump to 32 if int8 sanity fails

    cmd = [
        sys.executable,
        str(REPO_ROOT / 'scripts/edge/run_edge_eval_macbook.py'),
        '--config',     str(REPO_ROOT / 'configs/macbook.yaml'),
        '--output-dir', str(EDGE_RESULTS),
        '--calibration-batches', str(CALIBRATION_BATCHES),
        '--pilot-only',
    ]
    print('Running pilot:')
    print('  ' + ' \\\\\\n  '.join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f'Pilot run failed with exit code {result.returncode}. '
            'Inspect the log above before continuing.'
        )

    # Load and pretty-print the sanity report
    import json
    sanity_path = EDGE_RESULTS / 'results' / 'sanity_report.json'
    if sanity_path.is_file():
        sanity = json.loads(sanity_path.read_text())
        print('\\n=== Sanity Report ===')
        all_passed = True
        for row in sanity:
            mark = '✅' if row['passed'] else '❌'
            print(
                f'{mark} {row[\"runtime\"]:>7s} {row[\"mode\"]:>5s}  '
                f'max_abs={row[\"max_abs_diff\"]:.2e}  '
                f'cos={row[\"cosine_similarity\"]:.4f}  '
                f'threshold={row[\"threshold\"]:.2e}'
            )
            all_passed = all_passed and row['passed']
        if not all_passed:
            print('\\n⚠ At least one sanity check FAILED. Do NOT proceed to Cell 8.')
            print('   Re-run this cell with CALIBRATION_BATCHES = 32 first.')
            raise RuntimeError('Sanity check did not fully pass.')
        else:
            print('\\n✅ All sanity checks passed — green light for Cell 8 (mass eval).')
""").lstrip()))


# =============================================================================
# CELL 8 — Mass conversion + eval (the long one)
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 8 — Mass conversion + AUC evaluation (long-running)

    **Purpose:** Convert all 9 checkpoints to all 4 formats (or just TFLite if
    not on Apple Silicon), evaluate each on gen3 full test set + gen1/gen2 5K
    subsets, and benchmark latency.

    **Resume:** Uses `--resume` and per-checkpoint JSON caching. If your MacBook
    crashes/sleeps/reboots, just re-run THIS CELL — work that's already done is
    skipped via `per_ckpt_<run_tag>.json` files in
    `<EDGE_RESULTS>/results/`.

    **Expected output structure (in `EDGE_RESULTS/results/`):**
    ```
    per_ckpt_pytorch_baseline_gen3seed0.json
    per_ckpt_gen1_seed0.json
    per_ckpt_gen1_seed1.json
    per_ckpt_gen1_seed2.json
    per_ckpt_gen2_replay+ewc_seed0.json
    per_ckpt_gen2_replay+ewc_seed1.json
    per_ckpt_gen2_replay+ewc_seed2.json
    per_ckpt_gen3_replay+ewc_seed0.json
    per_ckpt_gen3_replay+ewc_seed1.json
    per_ckpt_gen3_replay+ewc_seed2.json
    edge_eval_full.json            ← final aggregate
    edge_eval_summary.md           ← thesis-ready markdown
    sanity_report.json             ← from Cell 7
    ```

    **Time:** ~2.5-3 hours on M2 Max:
    - ~30-50 min: mass conversion (8 checkpoints — pilot already done)
    - ~10 min: PyTorch CPU baseline (gen3_seed0 only)
    - ~60-90 min: edge AUC eval (9 ckpt × up to 4 fmt × 3 splits)
    - ~5 min: latency benchmark (pilot only, 4 fmt × multiple compute units)
    - ~1 min: aggregation + summary write

    **Tips for the wait:**
    - Don't close the MacBook lid — it'll sleep and pause the run
    - You can leave it overnight — the per-ckpt cache means partial progress
      is safe even if power dies
    - Monitor progress via the per_ckpt JSON count: `ls EDGE_RESULTS/results/per_ckpt_*.json | wc -l`
""").lstrip()))

CELLS.append(code(dedent("""
    import subprocess, sys, time

    # Tunable knobs (defaults are sensible for M2 Max)
    SUBSET_ROWS = 5000          # gen1/gen2 test subsample size
    NUM_LATENCY_RUNS = 200      # iterations per latency measurement
    NUM_WARMUP = 20             # warmup iterations
    BATCH_SIZE = 32             # eval batch size (lower if RAM tight)
    NUM_WORKERS = 2             # dataloader workers
    CALIBRATION_BATCHES = 16    # match Cell 7

    cmd = [
        sys.executable,
        str(REPO_ROOT / 'scripts/edge/run_edge_eval_macbook.py'),
        '--config',          str(REPO_ROOT / 'configs/macbook.yaml'),
        '--output-dir',      str(EDGE_RESULTS),
        '--subset-rows',     str(SUBSET_ROWS),
        '--num-latency-runs', str(NUM_LATENCY_RUNS),
        '--num-warmup',      str(NUM_WARMUP),
        '--batch-size',      str(BATCH_SIZE),
        '--num-workers',     str(NUM_WORKERS),
        '--calibration-batches', str(CALIBRATION_BATCHES),
        '--resume',
    ]
    print('Running full edge evaluation:')
    print('  ' + ' \\\\\\n  '.join(cmd))
    print('\\nThis will take ~2-3 hours. Per-checkpoint progress saved automatically.\\n')

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed_min = (time.time() - start) / 60

    if result.returncode != 0:
        print(f'\\n⚠ Run exited with code {result.returncode} after {elapsed_min:.1f} min.')
        print('   The per-checkpoint cache may have partial progress.')
        print('   Re-run this cell with --resume to continue from where it stopped.')
        raise RuntimeError('Mass eval did not complete.')
    print(f'\\n✅ Mass evaluation complete in {elapsed_min:.1f} min ({elapsed_min/60:.2f} hours)')
""").lstrip()))


# =============================================================================
# CELL 9 — Progress check / inspection
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 9 — Inspect per-checkpoint progress

    **Purpose:** Independent of Cell 8's exit code, look at what per-checkpoint
    JSONs exist on disk. Useful for:
    - Confirming all 10 expected files are present (9 ckpts + 1 baseline)
    - Spotting any checkpoint that's missing rows (failed format conversion)
    - Quick sanity check on the AUC numbers before they hit the thesis

    **Expected output:** A table showing one row per (run_tag, runtime, mode),
    with AUC + log_loss + accuracy across the 3 test splits.
""").lstrip()))

CELLS.append(code(dedent("""
    import json
    import pandas as pd
    from pathlib import Path

    per_ckpt_files = sorted((EDGE_RESULTS / 'results').glob('per_ckpt_*.json'))
    print(f'Found {len(per_ckpt_files)} per-checkpoint JSON file(s):')
    for f in per_ckpt_files:
        n_rows = len(json.loads(f.read_text()).get('rows', []))
        print(f'   {f.name}: {n_rows} eval row(s)')

    if not per_ckpt_files:
        print('\\n(no per_ckpt files yet — wait for Cell 8 to make progress)')
    else:
        # Build a quick summary dataframe
        all_rows = []
        for f in per_ckpt_files:
            payload = json.loads(f.read_text())
            all_rows.extend(payload.get('rows', []))
        df = pd.DataFrame(all_rows)
        if not df.empty:
            # Pivot to (run_tag, runtime, mode) × test_split → auc
            pivot = df.pivot_table(
                index=['run_tag', 'runtime', 'mode'],
                columns='test_split',
                values='auc',
                aggfunc='first',
            ).round(4)
            print('\\n=== AUC pivot ===')
            print(pivot.to_string())
""").lstrip()))


# =============================================================================
# CELL 10 — Latency table
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 10 — Latency benchmark table

    **Purpose:** Pretty-print the latency table from the run. This is the
    headline edge-deployment number for the thesis.

    **Expected output (rough ballpark on M2 Max):**

    | runtime | mode | compute_unit | mean (ms) | p95 (ms) | size (MB) |
    |---|---|---|---|---|---|
    | tflite | fp32 | cpu_xnnpack | ~3-5 | ~5-8  | ~18 |
    | tflite | int8 | cpu_xnnpack | ~1-3 | ~3-5  | ~5  |
    | coreml | fp32 | all (ANE+GPU) | ~1-3 | ~3-5 | ~18 |
    | coreml | int8 | all (ANE+GPU) | ~1-2 | ~2-4 | ~5  |
    | coreml | fp32 | cpu_only | similar to TFLite fp32 | | |
    | coreml | int8 | cpu_only | similar to TFLite int8 | | |
""").lstrip()))

CELLS.append(code(dedent("""
    import json
    import pandas as pd
    from pathlib import Path

    full_json = EDGE_RESULTS / 'results' / 'edge_eval_full.json'
    if not full_json.is_file():
        print(f'⚠ {full_json} not found yet. Cell 8 must complete first.')
    else:
        data = json.loads(full_json.read_text())
        lat = pd.DataFrame(data.get('latency', []))
        if lat.empty:
            print('No latency rows in the final JSON.')
        else:
            cols = ['runtime', 'mode', 'compute_unit', 'latency_ms_mean',
                    'latency_ms_p50', 'latency_ms_p95', 'latency_ms_p99', 'size_mb']
            print('=== Latency (M2 Max, single-image, batch=1) ===')
            print(lat[cols].round(2).to_string(index=False))
""").lstrip()))


# =============================================================================
# CELL 11 — AUC mean ± std summary
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 11 — AUC mean ± std (thesis numbers)

    **Purpose:** Aggregate the per-checkpoint AUC across seeds, grouped by
    (checkpoint generation, runtime, mode, test split). This is what goes
    into the thesis Section X.X table.

    **Expected output:** A grouped dataframe with mean / std / n=3 per group.
""").lstrip()))

CELLS.append(code(dedent("""
    import json
    import pandas as pd

    full_json = EDGE_RESULTS / 'results' / 'edge_eval_full.json'
    if not full_json.is_file():
        print(f'⚠ {full_json} not found yet.')
    else:
        data = json.loads(full_json.read_text())
        ev = pd.DataFrame(data.get('evaluation', []))
        if ev.empty:
            print('No evaluation rows.')
        else:
            # Extract checkpoint generation from run_tag
            ev['ckpt_gen'] = ev['run_tag'].str.extract(r'^(gen\\d)')[0]
            agg = (
                ev.groupby(['ckpt_gen', 'runtime', 'mode', 'test_split'])['auc']
                .agg(['mean', 'std', 'count'])
                .round(4)
                .reset_index()
            )
            print('=== AUC mean ± std by group ===')
            print(agg.to_string(index=False))
""").lstrip()))


# =============================================================================
# CELL 12 — Generate figures
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 12 — Generate figures

    **Purpose:** Produce two figures for the thesis:
    1. **Latency vs Size scatter** — quantization trade-off across formats
    2. **AUC degradation bar chart** — fp32 → int8 cost per generation

    Both saved as PNG (300 DPI, publication-quality) into
    `EDGE_RESULTS/results/figures/`.
""").lstrip()))

CELLS.append(code(dedent("""
    import json
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    fig_dir = EDGE_RESULTS / 'results' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    full_json = EDGE_RESULTS / 'results' / 'edge_eval_full.json'
    data = json.loads(full_json.read_text())

    # --- Figure 1: Latency vs Size ---
    lat = pd.DataFrame(data.get('latency', []))
    if not lat.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {
            ('tflite', 'fp32'): '#1f77b4',
            ('tflite', 'int8'): '#aec7e8',
            ('coreml', 'fp32'): '#ff7f0e',
            ('coreml', 'int8'): '#ffbb78',
        }
        markers = {'all': 'o', 'cpu_only': 's', 'cpu_xnnpack': 'D'}
        for _, row in lat.iterrows():
            ax.errorbar(
                row['size_mb'], row['latency_ms_mean'],
                yerr=[[row['latency_ms_mean'] - row['latency_ms_p50']],
                      [row['latency_ms_p95'] - row['latency_ms_mean']]],
                fmt=markers.get(row['compute_unit'], 'o'),
                color=colors.get((row['runtime'], row['mode']), 'gray'),
                markersize=10, capsize=4,
                label=f\"{row['runtime']}/{row['mode']}/{row['compute_unit']}\",
            )
        ax.set_xlabel('Model size (MB)', fontsize=12)
        ax.set_ylabel('Latency (ms, mean ± p50/p95)', fontsize=12)
        ax.set_title('Edge Deployment: Latency vs Model Size (Apple M2 Max)', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        plt.tight_layout()
        f1 = fig_dir / 'latency_vs_size.png'
        plt.savefig(f1, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'✅ Figure 1: {f1}')

    # --- Figure 2: AUC degradation fp32 → int8 ---
    ev = pd.DataFrame(data.get('evaluation', []))
    if not ev.empty:
        ev['ckpt_gen'] = ev['run_tag'].str.extract(r'^(gen\\d)')[0]
        # Focus on gen3 checkpoints (deployment-target model) × gen3 test split
        focus = ev[(ev['ckpt_gen'] == 'gen3') & (ev['test_split'] == 'gen3_test')]
        agg = (
            focus.groupby(['runtime', 'mode'])['auc']
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        if not agg.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [f\"{r['runtime']}\\n{r['mode']}\" for _, r in agg.iterrows()]
            x = range(len(agg))
            ax.bar(x, agg['mean'], yerr=agg['std'], capsize=6, alpha=0.8,
                   color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'][:len(agg)])
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.set_ylabel('AUC on gen3_test (mean ± std, n=3 seeds)', fontsize=12)
            ax.set_title('Quantization AUC Cost — gen3 Final Model on gen3_test', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0.80, 0.92)
            for i, (mean, std) in enumerate(zip(agg['mean'], agg['std'])):
                ax.text(i, mean + (std if pd.notna(std) else 0) + 0.002,
                        f'{mean:.4f}', ha='center', fontsize=9)
            plt.tight_layout()
            f2 = fig_dir / 'auc_degradation.png'
            plt.savefig(f2, dpi=300, bbox_inches='tight')
            plt.show()
            print(f'✅ Figure 2: {f2}')

    print(f'\\nFigures saved to: {fig_dir}')
""").lstrip()))


# =============================================================================
# CELL 13 — Preview thesis-ready summary
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 13 — Preview the thesis-ready Markdown summary

    **Purpose:** Display the Markdown summary that the orchestrator wrote.
    This is exactly what you'll paste into the thesis (or share back via chat
    for further polishing into Indonesian narrative).

    **Expected output:** ~50-100 lines of Markdown with sanity table, latency
    table, AUC mean ± std table, and full per-checkpoint appendix.
""").lstrip()))

CELLS.append(code(dedent("""
    md_path = EDGE_RESULTS / 'results' / 'edge_eval_summary.md'
    if not md_path.is_file():
        print(f'⚠ {md_path} not found.')
    else:
        from IPython.display import Markdown, display
        text = md_path.read_text()
        print(f'Summary path: {md_path}')
        print(f'Size: {len(text) / 1024:.1f} KB ({len(text.splitlines())} lines)')
        print('=' * 60)
        # Render nicely in the notebook
        display(Markdown(text))
""").lstrip()))


# =============================================================================
# CELL 14 — Optional cleanup
# =============================================================================
CELLS.append(md(dedent("""
    ---
    ## Cell 14 — Optional: cleanup extracted frames to free disk

    **Purpose:** After eval, you don't need the 80GB of extracted DF40 frames
    anymore — the checkpoints, conversion artifacts, results JSON, and figures
    are all small (~1 GB total). This cell helps free disk space if you want
    to keep the MacBook lean.

    **What stays:**
    - `<EDGE_RESULTS>/conversions/` — TFLite + CoreML artifacts (~500 MB)
    - `<EDGE_RESULTS>/results/` — all JSONs + Markdown + figures (~5 MB)
    - `<OUTPUT_ROOT>/mirror/checkpoints/` — 9 student .pth files (~820 MB)

    **What gets removed (if you uncomment the cell below):**
    - `<OUTPUT_ROOT>/df40/` — extracted technique frames (~75 GB)
    - `<OUTPUT_ROOT>/df40_real/` — extracted real frames (~5 GB)

    **DO NOT RUN THIS** unless the eval has completed and you've already kept a
    copy of `edge_eval_summary.md`.
""").lstrip()))

CELLS.append(code(dedent("""
    # UNCOMMENT to actually delete the extracted frames
    # import shutil
    # for p in (OUTPUT_ROOT / 'df40', OUTPUT_ROOT / 'df40_real'):
    #     if p.is_dir():
    #         print(f'Removing {p} ...')
    #         shutil.rmtree(p)
    # print('✅ Cleanup done — only checkpoints + results remain.')
    print('Cleanup cell is intentionally commented out — uncomment the lines above to run.')
""").lstrip()))


# =============================================================================
# Final: write notebook
# =============================================================================
NB_OUTPUT = Path(__file__).resolve().parent / "08_edge_evaluation_macbook.ipynb"
NB_JSON = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB_OUTPUT.write_text(json.dumps(NB_JSON, indent=1), encoding="utf-8")
print(f"Notebook written: {NB_OUTPUT}")
print(f"Cells: {len(CELLS)}")
