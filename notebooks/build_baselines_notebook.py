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

=== DATA-SAFETY DESIGN (learned from the A6 24h-loss incident) ===

The earlier A6 run lost 24h because (a) results were only synced at the very
end, and (b) Drive FUSE buffers writes asynchronously, so files that "appeared"
on the mount were never uploaded before the runtime reset.

This notebook fixes both, with THREE layers:

1. **Per-stage persist.** Every training stage copies its checkpoint + metrics
   to Drive THE MOMENT it finishes — not at the end of the run.
2. **Forced flush.** After each persist, ``drive.flush_and_unmount()`` blocks
   until the Drive backend has actually received the bytes, then remounts.
   This is the explicit guarantee the A6 run lacked.
3. **Reset-proof reads + resume.** Chained stages read the previous
   checkpoint from the DRIVE path (survives a local NVMe wipe), and a Drive
   marker per stage lets a restart skip already-completed work.

Net effect: a mid-run reset can only lose the single in-progress stage —
never a completed milestone.
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

# --- Title + recovery doc ----------------------------------------------- #
CELLS.append(md(dedent("""
    # CKD — Baselines + Ablation (Minimal Balanced Scope)

    Runs the comparison baselines + the one sensitivity ablation for the thesis,
    on Colab A100. **~16 h total**, unattended-safe.

    | Stage | What | Seeds | Approx time |
    |---|---|---|---|
    | B1 | Ensemble teacher upper-bound eval | 0 (deterministic) | ~1 min |
    | B3 | Naive fine-tuning (no anti-forgetting) gen2->gen3 | 1 | ~5 h |
    | B4 | CL without KD (replay+ewc, hard labels) gen2->gen3 | 1 | ~5 h |
    | A3 | Replay buffer sweep {5,10,15,20%} | 1 | ~5 h |

    **Not run:** B2 (full retrain) + B5 (static KD) dropped; A1 (KD contribution)
    answered by B4 vs the full CKD pipeline; A2/A4 from literature.

    ## 🛟 Data safety (read once)

    Every training stage **persists to Drive the instant it finishes** and then
    **force-flushes** (`drive.flush_and_unmount()` blocks until the bytes reach
    the Drive backend, then remounts). Chained stages read the previous
    checkpoint from the **Drive path**, so a local NVMe wipe doesn't break the
    chain. A per-stage **marker** on Drive lets a restart skip completed work.

    **A mid-run reset can only lose the single in-progress stage — never a
    completed milestone.** This is the fix for the earlier A6 24h loss.

    ## 🔁 If the runtime resets mid-run (recovery procedure)

    1. Start a fresh runtime (A100).
    2. Re-run **Cell 1 → Cell 2 → Cell 3** (repo, mirror, gen1 staging).
       The mirror re-extract takes ~20 min; this is unavoidable after a wipe.
    3. Re-run the training cell (**Cell 5**). It detects Drive markers and
       **skips every stage already done**, resuming from the first incomplete
       one. No completed work is repeated.

    To FORCE a re-run of a stage, delete its marker on Drive under
    `CKD_Thesis/results/raw/baselines/.b{3,4}_*_done` (or, for A3, the
    `.completed_seed0` markers under `results/raw/ablation/A3/<variant>/`).
""").lstrip()))

# --- Cell 1: GPU + repo -------------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 1 — GPU check + repo setup
    **Expected:** `A100` in the GPU line; pip install completes.
    **If not A100:** Runtime -> Change runtime type -> A100.
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
        # FORCE the repo to exactly match origin/main so we never run stale
        # code (a plain `pull --ff-only` can silently skip if the local copy
        # diverged). The checkout is pristine on Colab, so reset --hard is safe.
        subprocess.run(['git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', 'main'], check=True)
        subprocess.run(['git', '-C', str(REPO), 'reset', '--hard', 'origin/main'], check=True)
    else:
        token = getpass.getpass('GitHub token: ').strip()
        url = f'https://tesisbright123-blip:{token}@github.com/tesisbright123-blip/ckd-deepfake.git'
        subprocess.run(['git', 'clone', '--depth', '1', url, str(REPO)], check=True)
        del token, url
    os.chdir(REPO)

    # ASSURANCE: print exactly which commit is loaded. Compare this against the
    # latest commit you expect (e.g. the selective-extraction fix).
    sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    subj = subprocess.run(['git', '-C', str(REPO), 'log', '-1', '--pretty=%s'],
                          capture_output=True, text=True).stdout.strip()
    print(f'>>> Loaded commit {sha}: {subj}')

    subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
    subprocess.run(['pip', 'install', '-e', '.', '-q'], check=True)
    print('Repo ready at', REPO)
""").lstrip()))

# --- Cell 2: Local mirror ------------------------------------------------ #
CELLS.append(md(dedent("""
    ## Cell 2 — Local NVMe mirror (all generations)
    Extract DF40 to local NVMe (147x faster than Drive FUSE). B3/B4 need
    gen1+gen2+gen3; A3 needs gen1+gen2. We set up `all`.

    Technique zips are read **directly from Drive** (no local copy) and only
    the CSV-referenced frames are written — so the local footprint is ~32 GB
    of frames + ~6 GB real/uniface, NOT the 76 GB of zips or the full corpus.

    **Watch the log:** each technique zip prints
    `[selective] <zip>: extracted N frames -> df40/<tech>` then
    `[done] <zip> (disk free: XX GB)` — disk free should stay healthy
    (not creep toward 0). **Expected last line:** `=== SETUP COMPLETE ===`,
    verify rows 100%.
    **Time:** ~15-20 min (reads only referenced frames). Idempotent via `--resume`.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys
    from pathlib import Path

    # PRE-FLIGHT GUARD: confirm the loaded script has the ROBUST selective
    # extractor (direct-from-Drive, tail-matching, no full-extract fallback).
    # _frames_suffix is unique to that fix, so its presence proves the latest
    # code is loaded. If this fails, Cell 1 ran stale code — re-run Cell 1
    # (it force-resets to origin/main), then re-run this cell.
    mirror_src = Path('/content/ckd-deepfake/scripts/00_setup_local_mirror.py').read_text()
    assert '_extract_zip_selective_direct' in mirror_src and 'def _frames_suffix' in mirror_src, (
        'Stale mirror script (missing the direct selective extractor). Re-run '
        'Cell 1 to pull the latest code, then re-run this cell.'
    )
    print('OK: robust direct-from-Drive selective extractor present.')

    rc = subprocess.run(
        [sys.executable, '-u', 'scripts/00_setup_local_mirror.py',
         '--generations', 'all', '--resume'],
        cwd='/content/ckd-deepfake',
    ).returncode
    if rc != 0:
        # The mirror's detailed log streams to stdout but Colab may collapse it.
        # Print the tail of the log FILE so the real cause is always visible.
        log_path = Path('/content/ckd-deepfake/runs/setup_local_mirror.log')
        print('\\n' + '=' * 70)
        print('MIRROR FAILED — last 60 log lines (the real cause is here):')
        print('=' * 70)
        if log_path.is_file():
            tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-60:]
            print('\\n'.join(tail))
        else:
            print(f'(log file not found at {log_path})')
        print('=' * 70)
        raise RuntimeError(
            'Local mirror setup failed. Look at the log tail just above for a '
            '"[FAIL] gen X — techniques MISSING: ..." or "Extraction failed for '
            '<zip>: ..." line. Safe to re-run this cell — finished zips skip.'
        )
    print('Local mirror ready.')
""").lstrip()))

# --- Cell 3: gen1 checkpoint into local mirror --------------------------- #
CELLS.append(md(dedent("""
    ## Cell 3 — Stage the gen1 checkpoint into the local mirror
    B3/B4/A3 all fine-tune *from gen1*. A3's orchestrator reads the previous
    checkpoint from the LOCAL mirror path, so we stage gen1 best.pth there.
    (This is the exact gap that failed the earlier A6 attempt.) B3/B4 instead
    read from the Drive path directly — see Cell 5.
    **Expected:** `Copied ... best.pth` (or `already present`).
""").lstrip()))
CELLS.append(code(dedent("""
    import shutil
    from pathlib import Path

    GEN1_DRIVE_CANDIDATES = [
        Path('/content/drive/MyDrive/CKD_Thesis/checkpoints/students/gen1_seed0/best.pth'),
        Path('/content/drive/MyDrive/CKD_Thesis/checkpoints/students/gen1/best.pth'),
    ]
    GEN1_DRIVE = next((p for p in GEN1_DRIVE_CANDIDATES if p.is_file()), None)
    if GEN1_DRIVE is None:
        raise FileNotFoundError(
            'No gen1 checkpoint on Drive (gen1_seed0/best.pth or gen1/best.pth).'
        )
    print('gen1 source on Drive:', GEN1_DRIVE)

    dst = Path('/content/ckd_local/checkpoints/students/gen1/best.pth')
    if dst.is_file() and dst.stat().st_size == GEN1_DRIVE.stat().st_size:
        print('gen1 already staged locally at', dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(GEN1_DRIVE, dst)
        print(f'Copied {GEN1_DRIVE} -> {dst} ({dst.stat().st_size/1e6:.1f} MB)')
""").lstrip()))

# --- Cell 4: B1 ---------------------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 4 — B1: Ensemble teacher upper bound (no GPU)
    Evaluates the teacher ensemble on each generation's test set, reusing the
    `ensemble.npy` soft labels already on Drive. Pure file read, seconds.
    **Expected:** three `B1 genX ensemble: AUC=...` lines + a JSON path.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, shutil, json
    from pathlib import Path

    rc = subprocess.run(
        [sys.executable, '-u', 'scripts/baselines/eval_ensemble_teacher.py',
         '--config', 'configs/local.yaml', '--generations', 'all'],
        cwd='/content/ckd-deepfake',
    ).returncode
    if rc != 0:
        print('B1 non-zero exit — inspect log; partial results may still be written.')

    src = Path('/content/ckd_local/results/raw/baselines/B1_ensemble_metrics.json')
    if src.is_file():
        dst = Path('/content/drive/MyDrive/CKD_Thesis/results/raw/baselines/B1_ensemble_metrics.json')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        # Force flush so B1 is guaranteed on Drive even if the next long cell resets.
        from google.colab import drive
        drive.flush_and_unmount(); drive.mount('/content/drive')
        print('B1 synced + flushed ->', dst)
        for g in json.loads(src.read_text())['generations']:
            print(f\"  {g['generation']}: ensemble AUC={g.get('ensemble', {}).get('auc')}\")
""").lstrip()))

# --- Cell 5: combined robust training (B3 + B4 + A3) --------------------- #
CELLS.append(md(dedent("""
    ## Cell 5 — Training: B3 + B4 + A3 (unattended, resumable, data-safe)

    Runs all three training jobs back-to-back so the runtime never sits idle
    between them. **Each stage persists to Drive and force-flushes immediately
    on completion**, and reads its previous checkpoint from the Drive path — so
    a reset never breaks the chain or loses a completed stage.

    **What runs (in order):**
    - B3 gen2 (method `none`, from gen1) → B3 gen3 (from B3 gen2)
    - B4 gen2 (replay+ewc, `--no-soft-labels`, from gen1) → B4 gen3 (from B4 gen2)
    - A3 buffer sweep (orchestrator with its own per-variant Drive sync + resume)

    **Resume:** re-running this cell after a reset skips every stage whose Drive
    marker exists. **Time:** ~15 h. Auto-disconnects once at the very end (after
    a final flush) to stop idle compute burn.

    **Watch for:** each stage prints `[persisted + flushed to Drive]` right after
    it finishes — that line is your proof the milestone is safe.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, shutil, time
    from pathlib import Path
    from google.colab import drive, runtime as colab_runtime

    REPO = '/content/ckd-deepfake'
    LOCAL = Path('/content/ckd_local')
    DRIVE = Path('/content/drive/MyDrive/CKD_Thesis')
    L_CKPT = LOCAL / 'checkpoints/students/baselines'
    L_RES  = LOCAL / 'results/raw/baselines'
    D_CKPT = DRIVE / 'checkpoints/students/baselines'
    D_RES  = DRIVE / 'results/raw/baselines'
    for d in (L_CKPT, L_RES, D_CKPT, D_RES):
        d.mkdir(parents=True, exist_ok=True)

    GEN1_DRIVE = next(
        p for p in [
            DRIVE / 'checkpoints/students/gen1_seed0/best.pth',
            DRIVE / 'checkpoints/students/gen1/best.pth',
        ] if p.is_file()
    )

    def _flush():
        \"\"\"Block until Drive backend has the bytes, then remount. The guarantee
        the A6 run lacked — after this returns, data is truly persisted.\"\"\"
        try:
            drive.flush_and_unmount()
        except Exception as e:
            print('  flush warning (continuing):', e)
        drive.mount('/content/drive')
        assert (DRIVE).is_dir(), 'Drive remount failed after flush!'

    def _done(marker):
        return (D_RES / marker).is_file()

    def _persist(ckpt_dir: Path, metrics_glob: str, marker: str):
        \"\"\"Copy stage checkpoint(s) + metrics to Drive, write marker, force flush.\"\"\"
        # checkpoints
        for p in Path(ckpt_dir).glob('*.pth'):
            dst = D_CKPT / Path(ckpt_dir).name / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        # metrics
        for p in L_RES.glob(metrics_glob):
            shutil.copyfile(p, D_RES / p.name)
        (D_RES / marker).write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
        _flush()
        print(f'  [persisted + flushed to Drive] marker={marker}')

    def _train(gen, method, prev_ckpt, out_name, extra=None):
        cmd = [
            sys.executable, '-u', 'scripts/05_continual_distillation.py',
            '--config', 'configs/local.yaml', '--generation', gen,
            '--method', method, '--seed', '0',
            '--previous-checkpoint', str(prev_ckpt),
            '--checkpoint-dir', str(L_CKPT / out_name),
            '--results-dir', str(L_RES),
            '--num-workers', '2',
        ] + (extra or [])
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            raise RuntimeError(f'{out_name} failed (rc={rc}).')

    t0 = time.time()

    # ======================= B3 — naive fine-tuning ======================= #
    if _done('.b3_gen2_done'):
        print('[skip] B3 gen2 (marker on Drive)')
    else:
        print('=== B3 gen2 (method none, from gen1) ===')
        _train('gen2', 'none', GEN1_DRIVE, 'B3_gen2')
        _persist(L_CKPT / 'B3_gen2', 'gen2_none_continual_metrics*.json', '.b3_gen2_done')

    if _done('.b3_gen3_done'):
        print('[skip] B3 gen3 (marker on Drive)')
    else:
        print('=== B3 gen3 (method none, from B3 gen2 on Drive) ===')
        _train('gen3', 'none', D_CKPT / 'B3_gen2/best.pth', 'B3_gen3')
        _persist(L_CKPT / 'B3_gen3', 'gen3_none_continual_metrics*.json', '.b3_gen3_done')

    # ======================= B4 — CL without KD =========================== #
    if _done('.b4_gen2_done'):
        print('[skip] B4 gen2 (marker on Drive)')
    else:
        print('=== B4 gen2 (replay+ewc, --no-soft-labels, from gen1) ===')
        _train('gen2', 'replay+ewc', GEN1_DRIVE, 'B4_gen2', extra=['--no-soft-labels'])
        for p in L_RES.glob('gen2_replay+ewc_continual_metrics*.json'):
            p.rename(p.with_name(p.name.replace('gen2_replay+ewc', 'B4_gen2_replay+ewc')))
        _persist(L_CKPT / 'B4_gen2', 'B4_gen2_replay+ewc_continual_metrics*.json', '.b4_gen2_done')

    if _done('.b4_gen3_done'):
        print('[skip] B4 gen3 (marker on Drive)')
    else:
        print('=== B4 gen3 (replay+ewc, --no-soft-labels, from B4 gen2 on Drive) ===')
        _train('gen3', 'replay+ewc', D_CKPT / 'B4_gen2/best.pth', 'B4_gen3', extra=['--no-soft-labels'])
        for p in L_RES.glob('gen3_replay+ewc_continual_metrics*.json'):
            p.rename(p.with_name(p.name.replace('gen3_replay+ewc', 'B4_gen3_replay+ewc')))
        _persist(L_CKPT / 'B4_gen3', 'B4_gen3_replay+ewc_continual_metrics*.json', '.b4_gen3_done')

    # ======================= A3 — buffer sweep ============================ #
    # The orchestrator does its own per-variant Drive sync (+resume markers),
    # so we just run it and force a final flush afterwards.
    print('=== A3 buffer sweep (seed 0) ===')
    rc = subprocess.run([
        sys.executable, '-u', 'scripts/06_ablation_study.py',
        '--config', 'configs/local.yaml',
        '--ablation', 'A3', '--seeds', '0',
        '--drive-sync-root', str(DRIVE),
        '--resume',
    ], cwd=REPO).returncode
    if rc != 0:
        print(f'A3 exited {rc} — re-run this cell to resume (completed variants skip).')
    _flush()
    print('  [A3 synced + flushed to Drive]')

    print(f'\\nAll training done in {(time.time()-t0)/3600:.2f} h.')
    print('Final flush before auto-disconnect...')
    try:
        drive.flush_and_unmount()
    except Exception as e:
        print('  final flush warning:', e)
    print('Auto-disconnect in 60s (interrupt to keep the runtime alive)...')
    time.sleep(60)
    colab_runtime.unassign()
""").lstrip()))

# --- Cell 6: collect summary --------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 6 — Collect results into a summary

    Reads the B1/B3/B4/A3 outputs from Drive and writes
    `baselines_ablation_summary.md`. Run this on a fresh runtime after Cell 5
    finishes (mount Drive first if needed). Send the summary back for thesis
    polishing.
""").lstrip()))
CELLS.append(code(dedent("""
    import json
    from pathlib import Path
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        pass

    DRIVE = Path('/content/drive/MyDrive/CKD_Thesis')
    BASE = DRIVE / 'results/raw/baselines'
    lines = []
    def w(s=''):
        lines.append(s); print(s)

    w('# Baselines + Ablation Summary'); w()

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

    def _chain(tag, gen3_glob):
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
        w(f\"_best_val_auc gen3 = {d.get('best_val_auc')}_\"); w()

    _chain('B3 (naive FT, no protection)', 'gen3_none_continual_metrics*.json')
    _chain('B4 (CL without KD = A1)', 'B4_gen3_replay+ewc_continual_metrics*.json')

    a3 = DRIVE / 'results/raw/ablation/A3_summary.json'
    if a3.is_file():
        d = json.loads(a3.read_text())
        w('## A3 — Buffer size sensitivity (gen2, seed 0)')
        w('| Variant | gen1 AUC after S2 | gen2 AUC after S2 |')
        w('|---|---|---|')
        for r in d.get('runs', []):
            if r.get('status') != 'ok':
                continue
            aa = r.get('auc_after') or {}
            w(f\"| {r['variant']} | {aa.get('gen1', 'n/a')} | {aa.get('gen2', 'n/a')} |\")
        w()

    w('## Reference — Main CKD pipeline (Replay+EWC, with KD)')
    w('gen3 final AUC per gen (3-seed mean): gen1=0.7139, gen2=0.6383, gen3=0.8695')
    w('CGRS (N-1, same-method peak) = 0.840 ± 0.013'); w()

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
