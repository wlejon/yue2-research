"""Exp 2a/2b: does the prompt move the meters, and where do the attributes live?

Corpus runs are corpus/exp2/<genre>_<axis><+|->_s<seed> and <genre>_neutral_s<seed>.

  features  meter.features per clip (cached in cache/features.json)
  prompt    per axis: mean (+pole − −pole) of each meter over the 8 pairs, and how
            many of the 8 pairs agree in sign
  seams     per axis and per seam key (semantic layer / NAR layer+eval / latent):
            direction = mean paired difference; consistency = mean cosine
            between the per-pair differences; LOGO = leave-one-genre-out sign
            accuracy of the held-out pairs projected on the direction from the
            other genres; d' = mean projected pair gap / sd of projections
            within a pole. Directions saved to cache/dirs/.

    .venv/Scripts/python attributes.py corpus/exp2
"""
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import meter
from prompts import AXES, GENRES

CACHE = Path("cache")
PAT = re.compile(r"^(?P<g>[a-z]+)_(?P<ax>[a-z]+)(?P<pole>[+-]?)_s(?P<seed>\d+)$")


def load_corpus(root):
    clips = {}
    for d in sorted(Path(root).iterdir()):
        m = PAT.match(d.name)
        if m and (d / "audio.f32").exists():
            clips[d.name] = dict(dir=d, g=m["g"], ax=m["ax"], pole=m["pole"], seed=int(m["seed"]))
    return clips


def features(clips):
    path = CACHE / "features.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    for name, c in clips.items():
        if name in have:
            continue
        r = meter.load_run(c["dir"])
        f = meter.features(r.audio)
        f["energy"] = meter.energy_score(f)
        f["lat_norm"] = float(np.linalg.norm(r.latents, axis=1).mean())
        f["seconds"] = r.audio.shape[1] / meter.SR
        have[name] = f
        print("features", name, json.dumps({k: round(v, 3) for k, v in f.items()}), flush=True)
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(have, indent=1))
    return have


def pairs(clips, axis):
    out = []
    for g in GENRES:
        for seed in sorted({c["seed"] for c in clips.values()}):
            a, b = f"{g}_{axis}+_s{seed}", f"{g}_{axis}-_s{seed}"
            if a in clips and b in clips:
                out.append((g, a, b))
    return out


def vectors(clips):
    """name -> {key: vector}; keys are ('semantic', L, -1), ('synthesize', L, e), ('latent', 0, -1)."""
    out = {}
    for name, c in clips.items():
        v = {}
        if (c["dir"] / "hidden_keys.json").exists():
            v.update(meter.hidden_means(c["dir"]))
        r = meter.load_run(c["dir"])
        v[("latent", 0, -1)] = r.latents.mean(0)
        v[("latent_std", 0, -1)] = r.latents.std(0)
        out[name] = v
    return out


def unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


def seam_stats(vecs, prs, key):
    diffs = np.stack([vecs[a][key] - vecs[b][key] for _, a, b in prs]).astype(np.float64)
    direction = diffs.mean(0)
    u = [unit(d) for d in diffs]
    cons = float(np.mean([u[i] @ u[j] for i, j in itertools.combinations(range(len(u)), 2)]))
    genres = sorted({g for g, _, _ in prs})
    hits = n = 0
    for hg in genres:
        train = np.stack([d for (g, _, _), d in zip(prs, diffs) if g != hg]).mean(0)
        for (g, _, _), d in zip(prs, diffs):
            if g == hg:
                hits += int(d @ train > 0)
                n += 1
    du = unit(direction)
    pa = np.array([vecs[a][key] @ du for _, a, _ in prs])
    pb = np.array([vecs[b][key] @ du for _, _, b in prs])
    sd = np.sqrt((pa.var() + pb.var()) / 2) + 1e-12
    dprime = float((pa - pb).mean() / sd)
    base = np.mean([np.linalg.norm(vecs[a][key]) for _, a, _ in prs])
    return direction, dict(consistency=cons, logo=hits / max(1, n), dprime=dprime,
                           rel_norm=float(np.linalg.norm(direction) / base))


def main(root):
    clips = load_corpus(root)
    feats = features(clips)
    report = {"prompt": {}, "seams": {}}
    keys_f = meter.FEATURE_KEYS + ["energy", "lat_norm", "seconds"]
    lines = ["| axis | " + " | ".join(keys_f) + " |", "|---|" + "---|" * len(keys_f)]
    for ax in AXES:
        prs = pairs(clips, ax)
        row = {}
        for k in keys_f:
            d = np.array([feats[a][k] - feats[b][k] for _, a, b in prs])
            row[k] = (float(d.mean()), int((d > 0).sum()), len(d))
        report["prompt"][ax] = row
        lines.append(f"| {ax} | " + " | ".join(f"{m:+.2f} ({p}/{n})" for m, p, n in row.values()) + " |")
    prompt_table = "\n".join(lines)
    print(prompt_table)

    vecs = vectors(clips)
    any_name = next(iter(vecs))
    keys = sorted(vecs[any_name].keys(), key=lambda k: (k[0], k[1], k[2]))
    (CACHE / "dirs").mkdir(parents=True, exist_ok=True)
    seam_lines = ["| axis | seam | layer | eval | consistency | LOGO | d' | rel_norm |", "|---|---|---|---|---|---|---|---|"]
    for ax in AXES:
        prs = pairs(clips, ax)
        for key in keys:
            if not all(key in vecs[a] and key in vecs[b] for _, a, b in prs):
                continue
            direction, st = seam_stats(vecs, prs, key)
            report["seams"][f"{ax}|{key[0]}|{key[1]}|{key[2]}"] = st
            direction.astype(np.float32).tofile(CACHE / "dirs" / f"{ax}_{key[0]}_L{key[1]}_e{key[2]}.f32")
            seam_lines.append(f"| {ax} | {key[0]} | {key[1]} | {key[2]} | {st['consistency']:.3f} | {st['logo']:.2f} | "
                              f"{st['dprime']:.2f} | {st['rel_norm']:.3f} |")
    seam_table = "\n".join(seam_lines)
    Path("tables").mkdir(exist_ok=True)
    Path("tables/exp2_prompt.md").write_text(prompt_table + "\n")
    Path("tables/exp2_seams.md").write_text(seam_table + "\n")
    Path("tables/exp2.json").write_text(json.dumps(report, indent=1))
    # cross-axis entanglement at each axis's best seam
    print(seam_table)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "corpus/exp2")
