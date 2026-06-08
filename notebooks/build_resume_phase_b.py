"""Generator for notebooks/10_resume_phase_b.ipynb.

Run from repo root:
    python notebooks/build_resume_phase_b.py

A focused "resume Phase B" notebook for when Phase A (B3-gen2, B4-gen2, A3)
is ALREADY done and on Drive, and only B3-gen3 + B4-gen3 remain.

It extracts ONLY what Phase B needs — gen2 + gen3 (full) + gen1 TEST (for
cross-gen evaluation) — skipping gen1 train/val entirely (~13 GB and ~5 min
saved). One big self-contained cell does everything: repo setup, extraction,
training, per-stage Drive persist+flush, and a final auto-disconnect. A
second cell collects the summary (run on a fresh runtime afterwards).

Peak disk ~104 GB (fits a 112 GB free-tier disk).
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

CELLS.append(md(dedent("""
    # CKD — Resume Phase B (B3-gen3 + B4-gen3 only)

    Use this when Phase A is **already done** (B3-gen2, B4-gen2, A3 all on
    Drive) and only the two gen3 jobs remain. It extracts just what Phase B
    needs — **gen2 + gen3 (full) + gen1 TEST** (for cross-gen eval) — not gen1
    train/val, so it's leaner and faster than the full notebook.

    **Cell 1 does everything in one shot:** repo setup → extraction → B3-gen3 →
    B4-gen3 → per-stage Drive persist+flush → auto-disconnect. Peak disk
    ~104 GB. If it resets mid-run, just re-run Cell 1 — finished stages skip
    via Drive markers and already-extracted frames skip too.

    **Cell 2** collects the final summary — run it on a fresh runtime after
    Cell 1 finishes.
""").lstrip()))

# --- Cell 1: everything --------------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 1 — Resume Phase B end-to-end
    Repo + extract (gen2+gen3+gen1test) + B3-gen3 + B4-gen3. ~5-6 h.
    **Expected milestones:** `>>> Loaded commit ...`, `gen1 TEST restored`,
    `[persisted + flushed] .b3_gen3_done`, `[persisted + flushed] .b4_gen3_done`,
    `ALL DONE`.
""").lstrip()))
CELLS.append(code(dedent("""
    import subprocess, sys, os, getpass, shutil, time, importlib.util, logging
    from pathlib import Path
    import pandas as pd

    # --- repo + drive ---
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    print('GPU:', gpu)
    from google.colab import drive, runtime as colab_runtime
    drive.mount('/content/drive')
    REPO = Path('/content/ckd-deepfake')
    if REPO.exists():
        subprocess.run(['git','-C',str(REPO),'fetch','--depth','1','origin','main'], check=True)
        subprocess.run(['git','-C',str(REPO),'reset','--hard','origin/main'], check=True)
    else:
        tok = getpass.getpass('GitHub token: ').strip()
        url = f'https://tesisbright123-blip:{tok}@github.com/tesisbright123-blip/ckd-deepfake.git'
        subprocess.run(['git','clone','--depth','1',url,str(REPO)], check=True); del tok, url
    os.chdir(REPO)
    sha = subprocess.run(['git','-C',str(REPO),'rev-parse','--short','HEAD'],
                         capture_output=True, text=True).stdout.strip()
    print('>>> Loaded commit', sha)
    subprocess.run(['pip','install','-q','-r','requirements.txt'], check=True)
    subprocess.run(['pip','install','-e','.','-q'], check=True)

    LOCAL = Path('/content/ckd_local'); DRIVE = Path('/content/drive/MyDrive/CKD_Thesis')
    DF40 = Path('/content/df40_local/df40')
    L_CKPT = LOCAL/'checkpoints/students/baselines'; L_RES = LOCAL/'results/raw/baselines'
    D_CKPT = DRIVE/'checkpoints/students/baselines'; D_RES = DRIVE/'results/raw/baselines'
    for d in (L_CKPT, L_RES, D_CKPT, D_RES): d.mkdir(parents=True, exist_ok=True)

    def _flush():
        try: drive.flush_and_unmount()
        except Exception as e: print('flush warn:', e)
        drive.mount('/content/drive'); assert DRIVE.is_dir()
    def _done(m): return (D_RES/m).is_file()

    # --- 1. extract gen2 + gen3 (full) ---
    print('\\n=== Extract gen2 + gen3 ===')
    rc = subprocess.run([sys.executable,'-u','scripts/00_setup_local_mirror.py',
                         '--generations','gen2,gen3','--resume'], cwd=str(REPO)).returncode
    if rc != 0:
        lg = REPO/'runs/setup_local_mirror.log'
        if lg.is_file(): print('\\n'.join(lg.read_text(errors='replace').splitlines()[-40:]))
        raise RuntimeError('gen2+gen3 extraction failed.')

    # --- 2. local gen1_test.csv + restore gen1 TEST frames (for cross-gen eval) ---
    print('=== Restore gen1 TEST (csv + frames) ===')
    splits_local = LOCAL/'datasets/splits'; splits_local.mkdir(parents=True, exist_ok=True)
    drive_g1 = DRIVE/'datasets/splits/gen1_test.csv'
    g1 = pd.read_csv(drive_g1)
    g1['face_path'] = g1['face_path'].str.replace(
        '/content/drive/MyDrive/CKD_Thesis/datasets/raw/', '/content/df40_local/', regex=False)
    g1.to_csv(splits_local/'gen1_test.csv', index=False)

    spec = importlib.util.spec_from_file_location('mirror', str(REPO/'scripts/00_setup_local_mirror.py'))
    mirror = importlib.util.module_from_spec(spec); spec.loader.exec_module(mirror)
    mlog = logging.getLogger('g1t'); logging.basicConfig(level=logging.WARNING)
    by_tech = {}
    for fp in g1['face_path'].astype(str):
        if '/df40/' not in fp: continue
        t,_,r = fp.split('/df40/',1)[1].partition('/')
        if r: by_tech.setdefault(t, set()).add(r)
    tbl = {t.lower(): t for t in by_tech}
    GEN1 = ['blendface','e4s','facedancer','faceswap','fsgan','inswap','mobileswap','simswap','uniface']
    ZIPBK = DRIVE/'datasets/raw/df40_zip_backup'
    LZ = Path('/content/df40_local_zips'); LZ.mkdir(parents=True, exist_ok=True)
    for stem in GEN1:
        src = ZIPBK/(stem+'.zip')
        if not src.is_file(): print('  missing zip', stem); continue
        tech = tbl.get(stem.lower())
        # already restored? sample-check one file
        if tech and by_tech.get(tech):
            s = next(iter(by_tech[tech]))
            if (DF40/tech/s).is_file(): continue
        lz = LZ/(stem+'.zip'); shutil.copyfile(src, lz)
        try:
            if stem == 'uniface': mirror._extract_uniface_normalized(lz, logger=mlog)
            elif tech: mirror._extract_zip_selective_direct(lz, DF40, tech, by_tech[tech], logger=mlog)
        finally: lz.unlink()
    print(f'  gen1 TEST ready (disk free {shutil.disk_usage(\"/content\").free/1e9:.1f} GB)')

    # --- 3. B3-gen3 + B4-gen3 ---
    def _train(gen, method, prev, out, extra=None):
        cmd = [sys.executable,'-u','scripts/05_continual_distillation.py',
               '--config','configs/local.yaml','--generation',gen,'--method',method,
               '--seed','0','--previous-checkpoint',str(prev),
               '--checkpoint-dir',str(L_CKPT/out),'--results-dir',str(L_RES),
               '--num-workers','2'] + (extra or [])
        if subprocess.run(cmd, cwd=str(REPO)).returncode != 0:
            raise RuntimeError(f'{out} failed.')
    def _persist(ckpt, glob, marker):
        for p in Path(ckpt).glob('*.pth'):
            dst = D_CKPT/Path(ckpt).name/p.name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        for p in L_RES.glob(glob): shutil.copyfile(p, D_RES/p.name)
        (D_RES/marker).write_text(time.strftime('%Y-%m-%d %H:%M:%S')); _flush()
        print(f'  [persisted + flushed] {marker}')

    t0 = time.time()
    if _done('.b3_gen3_done'):
        print('[skip] B3 gen3')
    else:
        print('\\n=== B3 gen3 (none, from B3_gen2) ===')
        _train('gen3','none', D_CKPT/'B3_gen2/best.pth','B3_gen3')
        _persist(L_CKPT/'B3_gen3','gen3_none_continual_metrics*.json','.b3_gen3_done')
    if _done('.b4_gen3_done'):
        print('[skip] B4 gen3')
    else:
        print('\\n=== B4 gen3 (replay+ewc, --no-soft-labels, from B4_gen2) ===')
        _train('gen3','replay+ewc', D_CKPT/'B4_gen2/best.pth','B4_gen3', extra=['--no-soft-labels'])
        for p in L_RES.glob('gen3_replay+ewc_continual_metrics*.json'):
            p.rename(p.with_name(p.name.replace('gen3_replay+ewc','B4_gen3_replay+ewc')))
        _persist(L_CKPT/'B4_gen3','B4_gen3_replay+ewc_continual_metrics*.json','.b4_gen3_done')

    print(f'\\nALL DONE in {(time.time()-t0)/3600:.2f} h.')
    try: drive.flush_and_unmount()
    except Exception as e: print('final flush warn:', e)
    print('Auto-disconnect in 60s (interrupt to keep runtime alive)...')
    time.sleep(60)
    colab_runtime.unassign()
""").lstrip()))

# --- Cell 2: collect ----------------------------------------------------- #
CELLS.append(md(dedent("""
    ## Cell 2 — Collect summary (fresh runtime, after Cell 1)
    Reads B1/B3/B4/A3 from Drive, writes `baselines_ablation_summary.md`.
""").lstrip()))
CELLS.append(code(dedent("""
    import json
    from pathlib import Path
    try:
        from google.colab import drive; drive.mount('/content/drive')
    except Exception:
        pass
    DRIVE = Path('/content/drive/MyDrive/CKD_Thesis'); BASE = DRIVE/'results/raw/baselines'
    lines = []
    def w(s=''):
        lines.append(s); print(s)
    w('# Baselines + Ablation Summary'); w()
    b1 = BASE/'B1_ensemble_metrics.json'
    if b1.is_file():
        d = json.loads(b1.read_text())
        w('## B1 — Ensemble teacher (upper bound)'); w('| Gen | AUC | acc | n |'); w('|---|---|---|---|')
        for g in d['generations']:
            e = g.get('ensemble') or {}; a = e.get('auc')
            w(f\"| {g['generation']} | {('%.4f'%a) if a is not None else 'n/a'} | {('%.4f'%e['accuracy']) if a is not None else '—'} | {g.get('num_samples')} |\")
        w()
    def _chain(tag, glob):
        h = list(BASE.glob(glob))
        if not h: w(f'_{tag}: not found_'); w(); return
        d = json.loads(h[0].read_text()); aa = d.get('auc_after', {})
        w(f'## {tag} — after gen3'); w('| Test gen | AUC |'); w('|---|---|')
        for k in ('gen1','gen2','gen3'):
            if k in aa: w(f'| {k} | {aa[k]:.4f} |')
        w()
    _chain('B3 (naive FT)', 'gen3_none_continual_metrics*.json')
    _chain('B4 (CL without KD = A1)', 'B4_gen3_replay+ewc_continual_metrics*.json')
    a3 = DRIVE/'results/raw/ablation/A3_summary.json'
    if a3.is_file():
        d = json.loads(a3.read_text())
        w('## A3 — Buffer size sensitivity'); w('| Variant | gen1 after S2 | gen2 after S2 |'); w('|---|---|---|')
        for r in d.get('runs', []):
            if r.get('status') == 'ok':
                aa = r.get('auc_after') or {}
                w(f\"| {r['variant']} | {aa.get('gen1','n/a')} | {aa.get('gen2','n/a')} |\")
        w()
    w('## Reference — Main CKD: CGRS (N-1, same-method peak) = 0.840 ± 0.013'); w()
    out = BASE/'baselines_ablation_summary.md'; out.write_text('\\n'.join(lines), encoding='utf-8')
    print('\\nWrote', out)
""").lstrip()))


NB = Path(__file__).resolve().parent / "10_resume_phase_b.ipynb"
NB.write_text(json.dumps({
    "cells": CELLS,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4, "nbformat_minor": 5,
}, indent=1), encoding="utf-8")
print(f"Notebook written: {NB}")
print(f"Cells: {len(CELLS)}")
