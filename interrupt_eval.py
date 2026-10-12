"""Exp 3 analysis: per interrupt condition, the change it makes over the 10 s
after the switch, the seam it leaves at the switch, and what it costs.

  d_energy / d_lufs / d_centroid / d_onset / d_perc   span [P, P+N) vs the base's span
  pre_mel     log-mel L1 vs base over the 4 s before the switch (should be ~0)
  switch_z    excess spectral flux at the switch in base-flux sd (> ~2 = a step)
  span_mel    log-mel L1 vs base over the span (how much changed at all)

    .venv/Scripts/python interrupt_eval.py folk rock
"""
import json
import sys
from pathlib import Path

import numpy as np

import meter

ROOT = Path("runs/exp3")


def main(args):
    # A leading runs/... argument is the run directory (interrupt.py --out);
    # its tables are named after it.
    global ROOT
    songs = args
    if args and args[0].startswith("runs/"):
        ROOT, songs = Path(args[0]), args[1:]
    lines = ["| song | condition | d_energy | d_lufs | d_centroid | d_onset | d_perc | d_crest | flat_x | span_mel | pre_mel | switch_z |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for song in songs:
        base = meter.load_run(ROOT / f"{song}_base")
        info = base.info
        P, N = info["P"], info["N"]
        t0, t1 = P / 25, (P + N) / 25
        fb = meter.features(meter.segment(base.audio, t0, t1))
        fb["energy"] = meter.energy_score(fb)
        lmb = meter.logmel(meter.segment(base.audio, t0 - 4, t1))
        fluxb = meter.spectral_flux(lmb)
        sd = fluxb.std()
        sw = 4 * 25 * 4                      # switch column (4 s in, 4 columns per frame)
        for d in sorted(ROOT.glob(f"{song}_*")):
            if d.name == f"{song}_base" or not (d / "audio.f32").exists():
                continue
            r = meter.load_run(d)
            f = meter.features(meter.segment(r.audio, t0, t1))
            f["energy"] = meter.energy_score(f)
            lm = meter.logmel(meter.segment(r.audio, t0 - 4, t1))
            n = min(lm.shape[1], lmb.shape[1])
            diff = np.abs(lm[:, :n] - lmb[:, :n]).mean(0)
            flux = meter.spectral_flux(lm[:, :n])
            row = dict(song=song, cond=d.name[len(song) + 1:], d_energy=f["energy"] - fb["energy"],
                       d_lufs=f["lufs"] - fb["lufs"], d_centroid=f["centroid_hz"] - fb["centroid_hz"],
                       d_onset=f["onset_rate"] - fb["onset_rate"], d_perc=f["perc_ratio"] - fb["perc_ratio"],
                       d_crest=f["crest_db"] - fb["crest_db"], flat_x=f["flatness"] / max(fb["flatness"], 1e-9),
                       span_mel=float(diff[sw:].mean()), pre_mel=float(diff[:sw - 8].mean()),
                       switch_z=float((flux[sw - 2:sw + 3].max() - fluxb[sw - 2:sw + 3].max()) / sd),
                       info=r.info)
            rows.append(row)
            lines.append(f"| {song} | {row['cond']} | {row['d_energy']:+.2f} | {row['d_lufs']:+.2f} | "
                         f"{row['d_centroid']:+.0f} | {row['d_onset']:+.2f} | {row['d_perc']:+.3f} | "
                         f"{row['d_crest']:+.2f} | {row['flat_x']:.2f} | "
                         f"{row['span_mel']:.2f} | {row['pre_mel']:.3f} | {row['switch_z']:+.2f} |")
    t = "\n".join(lines)
    print(t)
    name = "exp3_interrupt" if ROOT.name == "exp3" else ROOT.name + "_interrupt"
    Path(f"tables/{name}.md").write_text(t + "\n")
    Path(f"tables/{name}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
