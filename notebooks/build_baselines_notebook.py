"""Generator for notebooks/09_baselines_ablation.ipynb.

Run from repo root:
    python notebooks/build_baselines_notebook.py

Runs the "minimal balanced" baseline + ablation scope on a FREE-TIER Colab
disk (~112 GB), which cannot hold all three generations' frames at once
(~58 GB data + ~59 GB base > 112 GB).

KEY INSIGHT: no single job ever needs all three generations on disk —
  A3 / B4-gen2 : gen1 + gen2
  B3-gen2      : gen2
  B4-gen3      : gen2 + gen3   (gen3 replays from gen2, not gen1)
  B3-gen3      : gen3
So the run is split into two phases that each hold at most TWO generations
(peak ~99 GB, comfortably under 112 GB):

  Phase A (gen1+gen2 on disk): A3 sweep, B3-gen2, B4-gen2
  -- swap: delete gen1 frames, extract gen3 (now gen2+gen3 on disk) --
  Phase B (gen2+gen3 on disk): B3-gen3, B4-gen3

Every training stage persists its checkpoint+metrics to Drive and force-
flushes the moment it finishes, and reads previous checkpoints from the
Drive path — so a reset only loses the in-progress stage, never a finished
one. The two phases run in ONE session (no disconnect between them) because
a disconnect would wipe the extracted frames; a single auto-disconnect
fires at the very end.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

GEN1_TECHS = [
    "blendface", "e4s", "facedancer", "faceswap", "fsgan",
    "inswap", "mobileswap", "simswap", "uniface",
]


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


# Shared helper block injected into both training cells (self-contained).
_HELPERS = dedent("""
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
        # Block until the Drive backend has the bytes, then remount — the
        # guarantee a plain copy lacks. After this returns, data is persisted.
        try:
            drive.flush_and_unmount()
        except Exception as e:
            print('  flush warning (continuing):', e)
        drive.mount('/content/drive')
        assert DRIVE.is_dir(), 'Drive remount failed after flush!'

    def _done(marker):
        return (D_RES / marker).is_file()

    def _persist(ckpt_dir: Path, metrics_glob: str, marker: str):
        for p in Path(ckpt_dir).glob('*.pth'):
            dst = D_CKPT / Path(ckpt_dir).name / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
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
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            raise RuntimeError(f'{out_name} failed.')
""").strip("\n")


CELLS: list[dict] = []

# --- Title + recovery doc ----------------------------------------------- #
CELLS.append(md(dedent("""
    # CKD — Baselines + Ablation (free-tier disk, 2-phase)

    Runs B1 + B3 + B4 + A3 on a free-tier Colab A100. Because the 112 GB disk
    cannot hold all three generations at once, the run is split into two phases
    that each hold at most TWO generations:

    | Phase | On disk | Jobs |
    |---|---|---|
    | A | gen1 + gen2 | A3 buffer sweep · B3-gen2 · B4-gen2 |
    | (swap) | delete gen1, add gen3 | — |
    | B | gen2 + gen3 | B3-gen3 · B4-gen3 |

    B1 (ensemble upper bound) needs no frames at all. **Peak disk ~99 GB.**

    ## 🛟 Data safety
    Each training stage persists its checkpoint+metrics to Drive and
    force-flushes the instant it finishes; chained stages read previous
    checkpoints from the Drive path. A mid-run reset loses only the
    in-progress stage. **Both phases must run in one session** (a disconnect
    wipes the extracted frames), so there's a single auto-disconnect at the end.

    ## 🔁 If the runtime resets
    Re-run Cell 1 → 2 → 3, then the training cells. Finished stages are skipped
    via their Drive markers; only the extraction (~15 min/phase) repeats.
    Run cells top to bottom.
""").lstrip()))

# --- Cell 1: GPU + repo -------------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 1 — GPU check + repo setup
    **Expected:** `A100`; then `>>> Loaded commit <sha>: ...` so you can confirm
    the latest code is loaded.
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
        subprocess.run(['git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', 'main'], check=True)
        subprocess.run(['git', '-C', str(REPO), 'reset', '--hard', 'origin/main'], check=True)
    else:
        token = getpass.getpass('GitHub token: ').strip()
        url = f'https://tesisbright123-blip:{token}@github.com/tesisbright123-blip/ckd-deepfake.git'
        subprocess.run(['git', 'clone', '--depth', '1', url, str(REPO)], check=True)
        del token, url
    os.chdir(REPO)
    sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    subj = subprocess.run(['git', '-C', str(REPO), 'log', '-1', '--pretty=%s'],
                          capture_output=True, text=True).stdout.strip()
    print(f'>>> Loaded commit {sha}: {subj}')
    subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
    subprocess.run(['pip', 'install', '-e', '.', '-q'], check=True)
    print('Repo ready.')
""").lstrip()))

# --- Cell 2: extract gen1+gen2 ------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 2 — Extract gen1 + gen2 (Phase A data)

    Only gen1+gen2 frames are extracted now (~42 GB) — gen3 comes later, after
    gen1 is freed. Reads zips from Drive, copies each to local, extracts only
    the CSV-referenced frames (layout-agnostic), deletes the zip. **Watch:**
    `[selective] <zip>: extracted N frames` and `[done] ... (disk free: XX GB)`
    — disk free should stay > 10 GB. **Expected:** `=== SETUP COMPLETE ===`.
    **Time:** ~12-15 min. On failure the real cause is printed from the log.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys
    from pathlib import Path

    mirror_src = Path('/content/ckd-deepfake/scripts/00_setup_local_mirror.py').read_text()
    assert '_extract_zip_selective_direct' in mirror_src and 'def _member_tail' in mirror_src, (
        'Stale mirror script. Re-run Cell 1 to pull the latest code.'
    )
    print('OK: layout-agnostic selective extractor present.')

    rc = subprocess.run(
        [sys.executable, '-u', 'scripts/00_setup_local_mirror.py',
         '--generations', 'gen1,gen2', '--resume'],
        cwd='/content/ckd-deepfake',
    ).returncode
    if rc != 0:
        log = Path('/content/ckd-deepfake/runs/setup_local_mirror.log')
        print('\\n' + '=' * 70 + '\\nMIRROR FAILED — last 60 log lines:\\n' + '=' * 70)
        if log.is_file():
            print('\\n'.join(log.read_text(errors='replace').splitlines()[-60:]))
        print('=' * 70)
        raise RuntimeError('gen1+gen2 extraction failed — see log tail above.')
    print('gen1+gen2 ready.')
""").lstrip()))

# --- Cell 3: stage gen1 checkpoint --------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 3 — Stage the gen1 checkpoint locally (for A3)
    A3's orchestrator reads the previous checkpoint from the local mirror path.
    **Expected:** `Copied ... best.pth` (or `already staged`).
""").lstrip()))
CELLS.append(code(dedent("""
    import shutil
    from pathlib import Path
    GEN1_DRIVE = next((p for p in [
        Path('/content/drive/MyDrive/CKD_Thesis/checkpoints/students/gen1_seed0/best.pth'),
        Path('/content/drive/MyDrive/CKD_Thesis/checkpoints/students/gen1/best.pth'),
    ] if p.is_file()), None)
    if GEN1_DRIVE is None:
        raise FileNotFoundError('No gen1 checkpoint on Drive.')
    dst = Path('/content/ckd_local/checkpoints/students/gen1/best.pth')
    if dst.is_file() and dst.stat().st_size == GEN1_DRIVE.stat().st_size:
        print('gen1 already staged:', dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(GEN1_DRIVE, dst)
        print(f'Copied {GEN1_DRIVE} -> {dst}')
""").lstrip()))

# --- Cell 4: B1 ---------------------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 4 — B1: Ensemble teacher upper bound (no frames, no GPU)
    Uses the `ensemble.npy` soft labels already on Drive. **Expected:** three
    `genX: ensemble AUC=...` lines.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, shutil, json
    from pathlib import Path
    subprocess.run([sys.executable, '-u', 'scripts/baselines/eval_ensemble_teacher.py',
                    '--config', 'configs/local.yaml', '--generations', 'all'],
                   cwd='/content/ckd-deepfake')
    src = Path('/content/ckd_local/results/raw/baselines/B1_ensemble_metrics.json')
    if src.is_file():
        dst = Path('/content/drive/MyDrive/CKD_Thesis/results/raw/baselines/B1_ensemble_metrics.json')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        from google.colab import drive
        drive.flush_and_unmount(); drive.mount('/content/drive')
        for g in json.loads(src.read_text())['generations']:
            print(f\"  {g['generation']}: ensemble AUC={g.get('ensemble', {}).get('auc')}\")
""").lstrip()))

# --- Cell 5: Phase A training -------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 5 — Phase A training (gen1+gen2 on disk): B3-gen2, B4-gen2, A3

    Runs the three Phase-A jobs. Each persists+flushes to Drive on completion.
    **No auto-disconnect** — Phase B (Cell 7) must run in the same session
    (a disconnect would wipe the frames). **Time:** ~10 h.
    Re-run to resume; finished stages skip via Drive markers.
""").lstrip()))
CELLS.append(code(_HELPERS + "\n" + dedent("""

    t0 = time.time()

    # B3 gen2 (naive, from gen1)
    if _done('.b3_gen2_done'):
        print('[skip] B3 gen2')
    else:
        print('=== B3 gen2 (method none) ===')
        _train('gen2', 'none', GEN1_DRIVE, 'B3_gen2')
        _persist(L_CKPT / 'B3_gen2', 'gen2_none_continual_metrics*.json', '.b3_gen2_done')

    # B4 gen2 (replay+ewc, hard labels, from gen1)
    if _done('.b4_gen2_done'):
        print('[skip] B4 gen2')
    else:
        print('=== B4 gen2 (replay+ewc, --no-soft-labels) ===')
        _train('gen2', 'replay+ewc', GEN1_DRIVE, 'B4_gen2', extra=['--no-soft-labels'])
        for p in L_RES.glob('gen2_replay+ewc_continual_metrics*.json'):
            p.rename(p.with_name(p.name.replace('gen2_replay+ewc', 'B4_gen2_replay+ewc')))
        _persist(L_CKPT / 'B4_gen2', 'B4_gen2_replay+ewc_continual_metrics*.json', '.b4_gen2_done')

    # A3 buffer sweep (orchestrator handles its own per-variant Drive sync)
    if _done('.a3_done'):
        print('[skip] A3 sweep')
    else:
        print('=== A3 buffer sweep (seed 0) ===')
        rc = subprocess.run([
            sys.executable, '-u', 'scripts/06_ablation_study.py',
            '--config', 'configs/local.yaml', '--ablation', 'A3', '--seeds', '0',
            '--drive-sync-root', str(DRIVE), '--resume',
        ], cwd=REPO).returncode
        if rc != 0:
            print(f'A3 exited {rc} — re-run this cell to resume.')
        else:
            (D_RES / '.a3_done').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
            _flush()
            print('  [A3 persisted + flushed]')

    print(f'\\nPhase A done in {(time.time()-t0)/3600:.2f} h. Continue to Cell 6.')
""").lstrip()))

# --- Cell 6: swap gen1 -> gen3 ------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 6 — Swap: free gen1 frames, extract gen3 (now gen2+gen3 on disk)

    Deletes the gen1 technique frames (gen1 is no longer needed — B4-gen3
    replays from gen2, not gen1) and extracts gen3, keeping gen2. Peak disk
    stays ~99 GB. **Expected:** `freed gen1` then `=== SETUP COMPLETE ===`.
    **Time:** ~10 min.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, shutil
    from pathlib import Path

    GEN1_TECHS = {t.lower() for t in %r}
    df40 = Path('/content/df40_local/df40')
    freed = 0
    if df40.is_dir():
        for sub in df40.iterdir():
            if sub.is_dir() and sub.name.lower() in GEN1_TECHS:
                shutil.rmtree(sub, ignore_errors=True)
                freed += 1
    free_gb = shutil.disk_usage('/content').free / 1e9
    print(f'freed gen1: removed {freed} technique folders (disk free now {free_gb:.1f} GB)')

    rc = subprocess.run(
        [sys.executable, '-u', 'scripts/00_setup_local_mirror.py',
         '--generations', 'gen3', '--resume'],
        cwd='/content/ckd-deepfake',
    ).returncode
    if rc != 0:
        log = Path('/content/ckd-deepfake/runs/setup_local_mirror.log')
        print('\\n' + '=' * 70 + '\\nMIRROR FAILED — last 60 log lines:\\n' + '=' * 70)
        if log.is_file():
            print('\\n'.join(log.read_text(errors='replace').splitlines()[-60:]))
        raise RuntimeError('gen3 extraction failed — see log tail above.')
    print('gen2+gen3 ready for Phase B.')
""" % (GEN1_TECHS,)).lstrip()))

# --- Cell 7: Phase B training -------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 7 — Phase B training (gen2+gen3 on disk): B3-gen3, B4-gen3

    Final two jobs. Previous checkpoints (B3_gen2, B4_gen2) are read from Drive.
    Auto-disconnects at the very end after a final flush. **Time:** ~5 h.
""").lstrip()))
CELLS.append(code(_HELPERS + "\n" + dedent("""

    t0 = time.time()

    # B3 gen3 (naive, from B3_gen2 on Drive)
    if _done('.b3_gen3_done'):
        print('[skip] B3 gen3')
    else:
        print('=== B3 gen3 (method none, from B3 gen2) ===')
        _train('gen3', 'none', D_CKPT / 'B3_gen2/best.pth', 'B3_gen3')
        _persist(L_CKPT / 'B3_gen3', 'gen3_none_continual_metrics*.json', '.b3_gen3_done')

    # B4 gen3 (replay+ewc, hard labels, from B4_gen2 on Drive)
    if _done('.b4_gen3_done'):
        print('[skip] B4 gen3')
    else:
        print('=== B4 gen3 (replay+ewc, --no-soft-labels, from B4 gen2) ===')
        _train('gen3', 'replay+ewc', D_CKPT / 'B4_gen2/best.pth', 'B4_gen3', extra=['--no-soft-labels'])
        for p in L_RES.glob('gen3_replay+ewc_continual_metrics*.json'):
            p.rename(p.with_name(p.name.replace('gen3_replay+ewc', 'B4_gen3_replay+ewc')))
        _persist(L_CKPT / 'B4_gen3', 'B4_gen3_replay+ewc_continual_metrics*.json', '.b4_gen3_done')

    print(f'\\nALL DONE (Phase B in {(time.time()-t0)/3600:.2f} h).')
    try:
        drive.flush_and_unmount()
    except Exception as e:
        print('final flush warning:', e)
    print('Auto-disconnect in 60s (interrupt to keep runtime alive)...')
    time.sleep(60)
    colab_runtime.unassign()
""").lstrip()))

# --- Cell 8: collect ----------------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 8 — Collect results into a summary
    Run on a fresh runtime after Cell 7. Reads B1/B3/B4/A3 from Drive and writes
    `baselines_ablation_summary.md`. Send it back for thesis polishing.
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
        w('| Gen | Ensemble AUC | acc | n |'); w('|---|---|---|---|')
        for g in d['generations']:
            e = g.get('ensemble') or {}
            auc, acc = e.get('auc'), e.get('accuracy')
            if auc is None:
                w(f\"| {g['generation']} | (not available yet) | — | — |\")
            else:
                w(f\"| {g['generation']} | {auc:.4f} | {acc:.4f} | {g.get('num_samples')} |\")
        w()

    def _chain(tag, gen3_glob):
        hits = list(BASE.glob(gen3_glob))
        if not hits:
            w(f'_{tag}: not found_'); w(); return
        d = json.loads(hits[0].read_text()); aa = d.get('auc_after', {})
        w(f'## {tag} — after gen3')
        w('| Test gen | AUC |'); w('|---|---|')
        for k in ('gen1', 'gen2', 'gen3'):
            if k in aa: w(f'| {k} | {aa[k]:.4f} |')
        w();

    _chain('B3 (naive FT)', 'gen3_none_continual_metrics*.json')
    _chain('B4 (CL without KD = A1)', 'B4_gen3_replay+ewc_continual_metrics*.json')

    a3 = DRIVE / 'results/raw/ablation/A3_summary.json'
    if a3.is_file():
        d = json.loads(a3.read_text())
        w('## A3 — Buffer size sensitivity (gen2, seed 0)')
        w('| Variant | gen1 after S2 | gen2 after S2 |'); w('|---|---|---|')
        for r in d.get('runs', []):
            if r.get('status') == 'ok':
                aa = r.get('auc_after') or {}
                w(f\"| {r['variant']} | {aa.get('gen1','n/a')} | {aa.get('gen2','n/a')} |\")
        w()

    w('## Reference — Main CKD (Replay+EWC, with KD)')
    w('CGRS (N-1, same-method peak) = 0.840 ± 0.013'); w()
    out = BASE / 'baselines_ablation_summary.md'
    out.write_text('\\n'.join(lines), encoding='utf-8')
    print('\\nWrote', out)
""").lstrip()))


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
