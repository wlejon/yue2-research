"""Exp 1 analysis: how windowed NAR synthesis differs from the whole-song NAR.

For every song, the reference is the protocol render (one bidirectional chunk)
of the run's own semantic tokens. Each variant renders the same tokens; a
variant whose name carries `_W<n>` has window seams at multiples of n frames
(or at the frames listed in its run.json "seams").

    .venv/Scripts/python seams.py runs/exp1 [--ref syn_W0] [--out tables/exp1_seams.md]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import meter

COLS = 4  # log-mel columns per latent frame (hop 480)


def seams_of(run, frames):
    s = run.info.get("seams")
    if s is not None:
        return np.array([x for x in s if 0 < x < frames])
    m = re.search(r"_W(\d+)", run.name)
    w = int(m.group(1)) if m else 0
    if w <= 0:
        return np.array([], dtype=int)
    return np.arange(w, frames, w)


def frame_jump(z):
    return np.concatenate([[np.nan], np.linalg.norm(np.diff(z, axis=0), axis=1)])


def analyse(ref, var, lm_ref=None, flux_ref=None):
    T = min(len(ref.latents), len(var.latents))
    zr, zv = ref.latents[:T], var.latents[:T]
    seams = seams_of(var, T)
    rel = np.linalg.norm(zv - zr, axis=1) / (np.linalg.norm(zr, axis=1) + 1e-6)
    cos = (zv * zr).sum(1) / (np.linalg.norm(zv, axis=1) * np.linalg.norm(zr, axis=1) + 1e-9)
    jr, jv = frame_jump(zr), frame_jump(zv)
    med_j = np.nanmedian(jr)
    out = dict(frames=T, n_seams=len(seams), lat_rel=float(rel.mean()), lat_cos=float(cos.mean()))
    if len(seams):
        out["seam_jump"] = float(np.nanmean(jv[seams]) / med_j)          # variant jump at seams / typical jump
        out["seam_jump_ref"] = float(np.nanmean(jr[seams]) / med_j)      # what the reference does there
        # error vs distance to the nearest seam (frames)
        idx = np.arange(T)
        dist = np.min(np.abs(idx[:, None] - np.concatenate([[0], seams])[None, :]), axis=1)
        out["rel_at_seam"] = float(rel[dist <= 1].mean())
        out["rel_mid"] = float(rel[dist >= 8].mean()) if (dist >= 8).any() else float("nan")
    if var.audio is not None and lm_ref is not None:
        lm = meter.logmel(var.audio)
        n = min(lm.shape[1], lm_ref.shape[1])
        flux = meter.spectral_flux(lm[:, :n])
        fr = flux_ref[:n]
        sd = np.std(fr)
        mel_d = np.abs(lm[:, :n] - lm_ref[:, :n]).mean(0)
        out["mel_l1"] = float(mel_d.mean())
        if len(seams):
            cols = np.clip(seams * COLS, 1, n - 2)
            win = np.stack([cols - 1, cols, cols + 1])
            fv = flux[win].max(0)
            frr = fr[win].max(0)
            out["seam_flux_z"] = float(((fv - frr) / sd).mean())         # excess flux at the seam, in ref sd
            out["seam_flux_p95"] = float(np.percentile((fv - frr) / sd, 95))
            out["seam_flux_ref_pctl"] = float((fv[:, None] > fr[None, :]).mean())  # where the seam flux sits in ref's flux distribution
            colidx = np.arange(n)
            cd = np.min(np.abs(colidx[:, None] - np.concatenate([[0], cols])[None, :]), axis=1)
            out["mel_l1_seam"] = float(mel_d[cd <= 4].mean())
            out["mel_l1_mid"] = float(mel_d[cd >= 32].mean()) if (cd >= 32).any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ref", default="syn_W0")
    ap.add_argument("--out")
    ap.add_argument("--pattern", default=r"^(?P<song>[a-z]+)_(?P<var>.+)$")
    a = ap.parse_args()
    root = Path(a.root)
    runs = defaultdict(dict)
    for d in sorted(root.iterdir()):
        if not (d / "latents.f32").exists():
            continue
        m = re.match(a.pattern, d.name)
        if m:
            runs[m["song"]][m["var"]] = d
    rows = []
    for song, vs in runs.items():
        if a.ref not in vs:
            continue
        ref = meter.load_run(vs[a.ref])
        lm_ref = meter.logmel(ref.audio) if ref.audio is not None else None
        flux_ref = meter.spectral_flux(lm_ref) if lm_ref is not None else None
        for var, d in sorted(vs.items()):
            if var == a.ref:
                continue
            r = analyse(ref, meter.load_run(d), lm_ref, flux_ref)
            r.update(song=song, var=var)
            rows.append(r)
            print(json.dumps(r))
    keys = ["song", "var", "frames", "n_seams", "lat_cos", "lat_rel", "rel_at_seam", "rel_mid", "seam_jump",
            "seam_jump_ref", "mel_l1", "mel_l1_seam", "mel_l1_mid", "seam_flux_z", "seam_flux_p95", "seam_flux_ref_pctl"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(
            (f"{r[k]:.3f}" if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in keys) + " |")
    table = "\n".join(lines)
    print(table)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(table + "\n", encoding="utf-8")
        (Path(a.out).with_suffix(".json")).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
