"""Generator for notebooks/09_baselines_ablation.ipynb.

Run from repo root:
    python notebooks/build_baselines_notebook.py

Produces the Colab notebook that runs the "minimal balanced" baseline +
ablation scope decided for the thesis:

    B1 — Ensemble teacher direct eval (no GPU, ~1 min)
    B3 — Naive fine-tuning chain gen2->gen3 (method none), 1 seed
    B4 — CL without KD chain gen2->gen3 (replay+ewc, --no-soft-labels), 1 seed
    A3 — Replay buffer size sensitivity {5,10,15,20%} (replay+ewc), 1 seed

A1 is answered by B4 vs the full CKD pipeline (no separate run).
A2/A4 are dropped (justified from literature; future work).

Every long cell: per-stage Drive sync (survives Colab session reset) +
auto-disconnect grace period. The gen1 checkpoint is copied into the local
mirror up front to avoid the "previous checkpoint not found" failure that
bit the earlier A6 attempt.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


CELLS: list[dict] = []

# --- Title --------------------------------------------------------------- #
CELLS.append(md(dedent("""
    # CKD — Baselines + Ablation (Minimal Balanced Scope)

    Runs the comparison baselines and the one sensitivity ablation decided for
    the thesis. **All on Colab A100.** Total compute ~16 hours; each stage syncs
    to Drive immediately so a session reset never loses completed work.

    | Stage | What | Seeds | Approx time |
    |---|---|---|---|
    | B1 | Ensemble teacher upper-bound eval | 0 (deterministic) | ~1 min |
    | B3 | Naive fine-tuning (no anti-forgetting) gen2->gen3 | 1 | ~5 h |
    | B4 | CL without KD (replay+ewc, hard labels) gen2->gen3 | 1 | ~5 h |
    | A3 | Replay buffer sweep {5,10,15,20%} | 1 | ~5 h |

    **Not run here:** B2 (full retrain) and B5 (static KD) are dropped; A1 (KD
    contribution) is answered by **B4 vs the full CKD pipeline**, so no separate
    A1 run. A2/A4 justified from literature.

    **Recovery model:** Every training stage writes its checkpoint + metrics to
    Drive the moment it finishes (per-stage sync). If Colab resets, re-run the
    same cell — completed stages are detected and skipped.

    Run cells top to bottom. Each has a markdown header with purpose + expected
    output + what to do on failure.
""").lstrip()))

# --- Cell 1: GPU + repo -------------------------------------------------- #
CELLS.append(md(dedent("""
    ## 1. GPU check + repo setup

    **Purpose:** confirm A100, clone/pull the repo, install deps.
    **Expected:** `A100` in the GPU line; pip install completes.
    **If not A100:** Runtime -> Change runtime type -> A100. B3/B4/A3 are slow on T4.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, os, getpass
    from pathlib import Path

    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    print('GPU:', gpu)
    if 'A100' not in gpu:
        print('WARNING: not A100 — training will be slow.')

    from google.colab import drive
    drive.mount('/content/drive')

    REPO = Path('/content/ckd-deepfake')
    if REPO.exists():
        subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
    else:
        token = getpass.getpass('GitHub token: ').strip()
        url = f'https://tesisbright123-blip:{token}@github.com/tesisbright123-blip/ckd-deepfake.git'
        subprocess.run(['git', 'clone', '--depth', '1', url, str(REPO)], check=True)
        del token, url
    os.chdir(REPO)
    subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
    subprocess.run(['pip', 'install', '-e', '.', '-q'], check=True)
    print('Repo ready at', REPO)
""").lstrip()))

# --- Cell 2: Local mirror ------------------------------------------------ #
CELLS.append(md(dedent("""
    ## 2. Local NVMe mirror (all generations)

    **Purpose:** extract DF40 to local NVMe (147x faster than Drive FUSE).
    B3/B4 need gen1+gen2+gen3; A3 needs gen1+gen2. We set up `all` to cover
    everything in one go.

    **Expected (last line):** `=== SETUP COMPLETE ===` with verify rows at 100%.
    **Time:** ~20-25 min. Idempotent via `--resume`.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys
    rc = subprocess.run(
        [sys.executable, '-u', 'scripts/00_setup_local_mirror.py',
         '--generations', 'all', '--resume'],
        cwd='/content/ckd-deepfake',
    ).returncode
    if rc != 0:
        raise RuntimeError('Local mirror setup failed — inspect the log above.')
    print('Local mirror ready.')
""").lstrip()))

# --- Cell 3: gen1 checkpoint into local mirror --------------------------- #
CELLS.append(md(dedent("""
    ## 3. Stage the gen1 checkpoint into the local mirror

    **Purpose:** B3/B4/A3 all fine-tune *from gen1*. The continual script loads
    the previous checkpoint from `{drive_root}/checkpoints/students/gen1/best.pth`
    where `drive_root` = the LOCAL mirror (`/content/ckd_local`). The mirror only
    symlinks teacher checkpoints, not student ones — so we copy gen1 best.pth in
    explicitly. (This is the exact gap that failed the earlier A6 attempt.)

    **Expected:** `Copied ... best.pth` (or `already present`).
""").lstrip()))
CELLS.append(code(dedent("""
    import shutil
    from pathlib import Path

    # Use the gen1 multi-seed checkpoint seed0 as the canonical gen1 start
    # (matches the main pipeline, whose S2/S3 also started from a gen1 seed0).
    src_candidates = [
        Path('/content/drive/MyDrive/CKD_Thesis/checkpoints/students/gen1_seed0/best.pth'),
        Path('/content/drive/MyDrive/CKD_Thesis/checkpoints/students/gen1/best.pth'),
    ]
    src = next((p for p in src_candidates if p.is_file()), None)
    if src is None:
        raise FileNotFoundError(
            'No gen1 checkpoint found on Drive (looked for gen1_seed0/best.pth '
            'and gen1/best.pth). Run initial distillation first.'
        )
    dst = Path('/content/ckd_local/checkpoints/students/gen1/best.pth')
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        print('gen1 checkpoint already present at', dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f'Copied {src} -> {dst} ({dst.stat().st_size/1e6:.1f} MB)')
""").lstrip()))

# --- Cell 4: B1 ---------------------------------------------------------- #
CELLS.append(md(dedent("""
    ## 4. B1 — Ensemble teacher upper bound (no GPU)

    **Purpose:** evaluate the teacher ensemble on each generation's test set,
    reusing the `ensemble.npy` soft labels already on Drive. This is the
    theoretical upper bound the student is distilled toward.

    **Expected:** three `B1 genX ensemble: AUC=...` lines, then a JSON path.
    **Time:** seconds (pure file read + metric compute, no GPU).
    **Note:** runs against `configs/local.yaml` (local mirror) — the ensemble.npy
    are symlinked from Drive so they're available.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, shutil
    from pathlib import Path

    rc = subprocess.run(
        [sys.executable, '-u', 'scripts/baselines/eval_ensemble_teacher.py',
         '--config', 'configs/local.yaml', '--generations', 'all'],
        cwd='/content/ckd-deepfake',
    ).returncode
    if rc != 0:
        print('B1 had a non-zero exit — inspect log; partial results may still be written.')

    # Sync B1 result to Drive
    src = Path('/content/ckd_local/results/raw/baselines/B1_ensemble_metrics.json')
    if src.is_file():
        dst = Path('/content/drive/MyDrive/CKD_Thesis/results/raw/baselines/B1_ensemble_metrics.json')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print('Synced B1 ->', dst)
        import json
        data = json.loads(src.read_text())
        for g in data['generations']:
            ens = g.get('ensemble', {})
            print(f\"  {g['generation']}: ensemble AUC={ens.get('auc')}\")
""").lstrip()))

# --- Cell 5: B3 ---------------------------------------------------------- #
CELLS.append(md(dedent("""
    ## 5. B3 — Naive fine-tuning (no anti-forgetting), gen2 -> gen3, 1 seed

    **Purpose:** quantify catastrophic forgetting with NO protection. Method
    `none` = plain sequential fine-tuning (KD still on, but no EWC / no replay).
    Expect gen1/gen2 AUC to collapse after gen3 (near-random) — that's the point.

    **Chain:** gen2 (from gen1) -> gen3 (from gen2-none). Custom output dirs keep
    these separate from the real CKD checkpoints.

    **Expected:** two training runs complete; metrics JSON written with `auc_after`
    showing big gen1/gen2 drops after gen3.
    **Time:** ~5 h (2 stages x ~2.5 h). Per-stage Drive sync; re-run to resume.
    **On failure:** re-run this cell — completed stages are skipped via the
    `.b3_done` markers on Drive.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, shutil, time
    from pathlib import Path
    from google.colab import runtime as colab_runtime

    LOCAL = Path('/content/ckd_local')
    DRIVE = Path('/content/drive/MyDrive/CKD_Thesis')
    CKPT_BASE = LOCAL / 'checkpoints/students/baselines'
    RES_BASE  = LOCAL / 'results/raw/baselines'
    DRIVE_BASE = DRIVE / 'results/raw/baselines'
    DRIVE_CKPT = DRIVE / 'checkpoints/students/baselines'
    for d in (CKPT_BASE, RES_BASE, DRIVE_BASE, DRIVE_CKPT):
        d.mkdir(parents=True, exist_ok=True)

    def _stage_done(marker_name):
        return (DRIVE_BASE / marker_name).is_file()

    def _sync_stage(ckpt_dir, res_dir, marker_name):
        # checkpoints
        for p in Path(ckpt_dir).glob('*.pth'):
            dst = DRIVE_CKPT / Path(ckpt_dir).name / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        # metrics
        for p in Path(res_dir).glob('*.json'):
            shutil.copyfile(p, DRIVE_BASE / p.name)
        (DRIVE_BASE / marker_name).write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
        print('  synced + marked', marker_name)

    start = time.time()

    # --- Stage B3-gen2 ---
    if _stage_done('.b3_gen2_done'):
        print('[skip] B3 gen2 already done (Drive marker)')
    else:
        print('=== B3 gen2 (method none) ===')
        rc = subprocess.run([
            sys.executable, '-u', 'scripts/05_continual_distillation.py',
            '--config', 'configs/local.yaml', '--generation', 'gen2',
            '--method', 'none', '--seed', '0',
            '--previous-checkpoint', str(LOCAL / 'checkpoints/students/gen1/best.pth'),
            '--checkpoint-dir', str(CKPT_BASE / 'B3_gen2'),
            '--results-dir', str(RES_BASE),
            '--num-workers', '2',
        ], cwd='/content/ckd-deepfake').returncode
        if rc != 0:
            raise RuntimeError('B3 gen2 failed.')
        _sync_stage(CKPT_BASE / 'B3_gen2', RES_BASE, '.b3_gen2_done')

    # --- Stage B3-gen3 ---
    if _stage_done('.b3_gen3_done'):
        print('[skip] B3 gen3 already done')
    else:
        print('=== B3 gen3 (method none, from B3 gen2) ===')
        rc = subprocess.run([
            sys.executable, '-u', 'scripts/05_continual_distillation.py',
            '--config', 'configs/local.yaml', '--generation', 'gen3',
            '--method', 'none', '--seed', '0',
            '--previous-checkpoint', str(CKPT_BASE / 'B3_gen2/best.pth'),
            '--checkpoint-dir', str(CKPT_BASE / 'B3_gen3'),
            '--results-dir', str(RES_BASE),
            '--num-workers', '2',
        ], cwd='/content/ckd-deepfake').returncode
        if rc != 0:
            raise RuntimeError('B3 gen3 failed.')
        _sync_stage(CKPT_BASE / 'B3_gen3', RES_BASE, '.b3_gen3_done')

    print(f'B3 done in {(time.time()-start)/60:.1f} min')
    print('Auto-disconnect in 60s (interrupt to keep runtime alive)...')
    time.sleep(60)
    colab_runtime.unassign()
""").lstrip()))

# --- Cell 6: B4 ---------------------------------------------------------- #
CELLS.append(md(dedent("""
    ## 6. B4 — CL without KD (replay+ewc, hard labels only), gen2 -> gen3, 1 seed

    **Purpose:** isolate the contribution of Knowledge Distillation. Same
    Replay+EWC machinery as the main pipeline, but trained with HARD LABELS ONLY
    (`--no-soft-labels`). The replay buffer also drops soft labels (fixed in the
    latest code) so the "no KD" condition is clean. **B4 vs the full CKD pipeline
    answers ablation A1.**

    **Expected:** two runs complete; B4 final gen3 AUC compared against CKD's
    0.87 will show the KD contribution (expect B4 lower if KD helps).
    **Time:** ~5 h. Per-stage Drive sync + resume via `.b4_*_done` markers.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, shutil, time
    from pathlib import Path
    from google.colab import runtime as colab_runtime

    LOCAL = Path('/content/ckd_local')
    DRIVE = Path('/content/drive/MyDrive/CKD_Thesis')
    CKPT_BASE = LOCAL / 'checkpoints/students/baselines'
    RES_BASE  = LOCAL / 'results/raw/baselines'
    DRIVE_BASE = DRIVE / 'results/raw/baselines'
    DRIVE_CKPT = DRIVE / 'checkpoints/students/baselines'

    def _stage_done(m): return (DRIVE_BASE / m).is_file()
    def _sync_stage(ckpt_dir, res_dir, m):
        for p in Path(ckpt_dir).glob('*.pth'):
            dst = DRIVE_CKPT / Path(ckpt_dir).name / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        for p in Path(res_dir).glob('*.json'):
            shutil.copyfile(p, DRIVE_BASE / p.name)
        (DRIVE_BASE / m).write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
        print('  synced + marked', m)

    start = time.time()

    if _stage_done('.b4_gen2_done'):
        print('[skip] B4 gen2 already done')
    else:
        print('=== B4 gen2 (replay+ewc, --no-soft-labels) ===')
        rc = subprocess.run([
            sys.executable, '-u', 'scripts/05_continual_distillation.py',
            '--config', 'configs/local.yaml', '--generation', 'gen2',
            '--method', 'replay+ewc', '--no-soft-labels', '--seed', '0',
            '--previous-checkpoint', str(LOCAL / 'checkpoints/students/gen1/best.pth'),
            '--checkpoint-dir', str(CKPT_BASE / 'B4_gen2'),
            '--results-dir', str(RES_BASE),
            '--num-workers', '2',
        ], cwd='/content/ckd-deepfake').returncode
        if rc != 0:
            raise RuntimeError('B4 gen2 failed.')
        # Rename metrics so they don't collide with the KD run's filename
        for p in RES_BASE.glob('gen2_replay+ewc_continual_metrics*.json'):
            p.rename(p.with_name(p.name.replace('gen2_replay+ewc', 'B4_gen2_replay+ewc')))
        _sync_stage(CKPT_BASE / 'B4_gen2', RES_BASE, '.b4_gen2_done')

    if _stage_done('.b4_gen3_done'):
        print('[skip] B4 gen3 already done')
    else:
        print('=== B4 gen3 (replay+ewc, --no-soft-labels, from B4 gen2) ===')
        rc = subprocess.run([
            sys.executable, '-u', 'scripts/05_continual_distillation.py',
            '--config', 'configs/local.yaml', '--generation', 'gen3',
            '--method', 'replay+ewc', '--no-soft-labels', '--seed', '0',
            '--previous-checkpoint', str(CKPT_BASE / 'B4_gen2/best.pth'),
            '--checkpoint-dir', str(CKPT_BASE / 'B4_gen3'),
            '--results-dir', str(RES_BASE),
            '--num-workers', '2',
        ], cwd='/content/ckd-deepfake').returncode
        if rc != 0:
            raise RuntimeError('B4 gen3 failed.')
        for p in RES_BASE.glob('gen3_replay+ewc_continual_metrics*.json'):
            p.rename(p.with_name(p.name.replace('gen3_replay+ewc', 'B4_gen3_replay+ewc')))
        _sync_stage(CKPT_BASE / 'B4_gen3', RES_BASE, '.b4_gen3_done')

    print(f'B4 done in {(time.time()-start)/60:.1f} min')
    print('Auto-disconnect in 60s (interrupt to keep runtime alive)...')
    time.sleep(60)
    colab_runtime.unassign()
""").lstrip()))

# --- Cell 7: A3 ---------------------------------------------------------- #
CELLS.append(md(dedent("""
    ## 7. A3 — Replay buffer size sensitivity {5,10,15,20%}, 1 seed

    **Purpose:** sensitivity analysis around the chosen 10% buffer. Uses the
    ablation orchestrator (`--ablation A3`) which runs gen2 with replay+ewc at
    each buffer fraction. 10% is one of the points so the curve is centred on the
    main-pipeline config.

    **Resume + Drive sync:** the orchestrator's `--drive-sync-root` + `--resume`
    (added after the A6 incident) copy each variant to Drive immediately and skip
    completed variants on re-run.

    **Expected:** 4 variants complete (buf_05/10/15/20pct), metrics under
    `results/raw/ablation/A3/`.
    **Time:** ~5 h (4 x ~75 min, but buf_10 may reuse less). Re-run to resume.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, time
    from google.colab import runtime as colab_runtime

    start = time.time()
    print('=== A3 buffer sweep (seed 0) ===')
    rc = subprocess.run([
        sys.executable, '-u', 'scripts/06_ablation_study.py',
        '--config', 'configs/local.yaml',
        '--ablation', 'A3', '--seeds', '0',
        '--drive-sync-root', '/content/drive/MyDrive/CKD_Thesis',
        '--resume',
    ], cwd='/content/ckd-deepfake').returncode
    if rc != 0:
        print(f'A3 exited {rc} — re-run this cell to resume (completed variants skip).')
    else:
        print(f'A3 done in {(time.time()-start)/60:.1f} min')

    print('Auto-disconnect in 60s (interrupt to keep runtime alive)...')
    time.sleep(60)
    colab_runtime.unassign()
""").lstrip()))

# --- Cell 8: collect + summarize ----------------------------------------- #
CELLS.append(md(dedent("""
    ## 8. Collect results into a summary

    **Purpose:** read the B1/B3/B4/A3 outputs from Drive and print a consolidated
    table for the thesis. Compares against the main CKD numbers (gen3 final AUC
    per generation) so the contribution of KD (B4) and anti-forgetting (B3) is
    visible at a glance.

    **Expected:** a printed table + a written `baselines_ablation_summary.md` on
    Drive. Send that file's contents back for thesis polishing.
""").lstrip()))
CELLS.append(code(dedent("""
    import json
    from pathlib import Path

    DRIVE = Path('/content/drive/MyDrive/CKD_Thesis')
    BASE = DRIVE / 'results/raw/baselines'
    lines = []
    def w(s=''):
        lines.append(s); print(s)

    w('# Baselines + Ablation Summary')
    w()

    # B1
    b1 = BASE / 'B1_ensemble_metrics.json'
    if b1.is_file():
        d = json.loads(b1.read_text())
        w('## B1 — Ensemble teacher (upper bound)')
        w('| Gen | Ensemble AUC | log_loss | acc | n |')
        w('|---|---|---|---|---|')
        for g in d['generations']:
            e = g.get('ensemble', {})
            w(f\"| {g['generation']} | {e.get('auc'):.4f} | {e.get('log_loss'):.4f} | {e.get('accuracy'):.4f} | {g.get('num_samples')} |\")
        w()

    # B3 / B4 cross-gen
    def _show_chain(tag, gen3_glob):
        hits = list(BASE.glob(gen3_glob))
        if not hits:
            w(f'_{tag}: gen3 metrics not found ({gen3_glob})_'); w(); return
        d = json.loads(hits[0].read_text())
        aa = d.get('auc_after', {})
        w(f'## {tag} — after gen3 (cross-gen retention)')
        w('| Test gen | AUC after S3 |')
        w('|---|---|')
        for k in ('gen1', 'gen2', 'gen3'):
            if k in aa: w(f'| {k} | {aa[k]:.4f} |')
        w(f\"_best_val_auc gen3 = {d.get('best_val_auc')}_\")
        w()

    _show_chain('B3 (naive FT, no protection)', 'gen3_none_continual_metrics*.json')
    _show_chain('B4 (CL without KD = A1)', 'B4_gen3_replay+ewc_continual_metrics*.json')

    # A3
    a3_summary = DRIVE / 'results/raw/ablation/A3_summary.json'
    if a3_summary.is_file():
        d = json.loads(a3_summary.read_text())
        w('## A3 — Buffer size sensitivity (gen2, seed 0)')
        w('| Variant | gen1 AUC after S2 | gen2 AUC after S2 |')
        w('|---|---|---|')
        for r in d.get('runs', []):
            if r.get('status') != 'ok':
                continue
            aa = r.get('auc_after') or {}
            w(f\"| {r['variant']} | {aa.get('gen1', 'n/a')} | {aa.get('gen2', 'n/a')} |\")
        w()

    # Reference: main CKD
    w('## Reference — Main CKD pipeline (Replay+EWC, with KD)')
    w('gen3 final AUC per gen (3-seed mean): gen1=0.7139, gen2=0.6383, gen3=0.8695')
    w('CGRS (N-1, same-method peak) = 0.840 ± 0.013')
    w()

    out = BASE / 'baselines_ablation_summary.md'
    out.write_text('\\n'.join(lines), encoding='utf-8')
    print('\\nWrote', out)
""").lstrip()))


# --- write notebook ------------------------------------------------------ #
NB_OUTPUT = Path(__file__).resolve().parent / "09_baselines_ablation.ipynb"
NB_JSON = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB_OUTPUT.write_text(json.dumps(NB_JSON, indent=1), encoding="utf-8")
print(f"Notebook written: {NB_OUTPUT}")
print(f"Cells: {len(CELLS)}")
