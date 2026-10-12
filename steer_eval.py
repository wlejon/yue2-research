"""Exp 2c: verify attribute directions by steering held-out prompts.

Runs in runs/exp2c: <genre>_base_s<seed>, <genre>_ar<L><+|-><scale>_s<seed> (semantic
stage steered, plan fixed), <genre>_nar<L><+|-><scale>_s<seed> (NAR steered on the
base's own tokens). This script adds <genre>_lat<+|-><scale>_s<seed>: the base
latents offset by scale * the latent direction (decoded with the reference VAE).

Per steered clip, versus its base: meter deltas, the energy score delta, and
the break detector (flatness x2, onset rate < half, LUFS < -40, latent norm out
of the corpus range).

    CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python steer_eval.py [axis] [lat scales]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import meter

ROOT = Path("runs/exp2c")
PAT = re.compile(r"^(?P<g>[a-z]+)_(?P<seam>base|ar|nar|lat)(?P<layer>\d*)(?P<scale>[+-][\d.]+)?_s(?P<seed>\d+)$")


def latent_renders(axis, scales):
    import torch
    from yue2.modeling_vae import YuE2VAE
    from lab import VAE_DIR, save_run
    vae = None
    d = np.fromfile(f"cache/dirs/{axis}_latent_L0_e-1.f32", dtype=np.float32)
    for base in sorted(ROOT.glob("*_base_s*")):
        r = meter.load_run(base)
        g, seed = base.name.split("_base_s")
        for s in scales:
            for sign in (1, -1):
                out = ROOT / f"{g}_lat{'+' if sign > 0 else '-'}{s:g}_s{seed}"
                if (out / "audio.f32").exists():
                    continue
                if vae is None:
                    vae = YuE2VAE.from_pretrained(VAE_DIR, decoder_only=True, device="cuda", local_files_only=True)
                z = r.latents + sign * s * d[None]
                with torch.inference_mode():
                    a = vae.decode_tiled(torch.from_numpy(z.T.copy())[None], core_frames=1024, halo_frames=16,
                                         output_device="cpu")[0].float().numpy()
                save_run(out, latents=z, audio=a, info={})


def main():
    axis = sys.argv[1] if len(sys.argv) > 1 else "energy"
    lat_scales = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.5,1,2,4").split(",")]
    latent_renders(axis, lat_scales)
    corpus = json.loads(Path("cache/features.json").read_text())
    ln = [f["lat_norm"] for f in corpus.values()]
    lat_lo, lat_hi = min(ln), max(ln)
    feats = {}
    for d in sorted(ROOT.iterdir()):
        m = PAT.match(d.name)
        if not m or not (d / "audio.f32").exists():
            continue
        r = meter.load_run(d)
        f = meter.features(r.audio)
        f["energy"] = meter.energy_score(f)
        f["lat_norm"] = float(np.linalg.norm(r.latents, axis=1).mean())
        feats[d.name] = (m, f)
    rows = defaultdict(list)
    for name, (m, f) in feats.items():
        if m["seam"] == "base":
            continue
        b = feats.get(f"{m['g']}_base_s{m['seed']}")
        if not b:
            continue
        fb = b[1]
        broken = (f["flatness"] > 2 * fb["flatness"] + 0.01 or f["onset_rate"] < 0.5 * fb["onset_rate"]
                  or f["lufs"] < -40 or not lat_lo * 0.8 <= f["lat_norm"] <= lat_hi * 1.2)
        rows[(m["seam"], m["layer"], float(m["scale"]))].append(dict(
            d_energy=f["energy"] - fb["energy"], d_lufs=f["lufs"] - fb["lufs"],
            d_centroid=f["centroid_hz"] - fb["centroid_hz"], d_onset=f["onset_rate"] - fb["onset_rate"],
            d_perc=f["perc_ratio"] - fb["perc_ratio"], flat_x=(f["flatness"] + 1e-4) / (fb["flatness"] + 1e-4),
            lat_norm=f["lat_norm"], broken=broken))
    keys = ["d_energy", "d_lufs", "d_centroid", "d_onset", "d_perc", "flat_x", "lat_norm"]
    lines = ["| seam | layer | scale | n | " + " | ".join(keys) + " | broken | energy up |",
             "|---|---|---|---|" + "---|" * len(keys) + "---|---|"]
    for (seam, layer, scale), rs in sorted(rows.items()):
        mean = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        up = sum(r["d_energy"] > 0 for r in rs)
        lines.append(f"| {seam} | {layer or '-'} | {scale:+g} | {len(rs)} | " +
                     " | ".join(f"{mean[k]:+.2f}" if k != "d_centroid" else f"{mean[k]:+.0f}" for k in keys) +
                     f" | {sum(r['broken'] for r in rs)} | {up}/{len(rs)} |")
    t = "\n".join(lines)
    print(t)
    Path("tables").mkdir(exist_ok=True)
    Path(f"tables/exp2c_steer_{axis}.md").write_text(t + "\n")


if __name__ == "__main__":
    main()
