"""Exp 4c analysis: mid-plan edits (from the folk plan's '% bridge' on) followed
by a fresh semantic stage with the same seed. Per edit: where its semantic
tokens part from the unedited plan's, and the tempo (median beat interval) and
energy before the split and over the song's last 35%, against the unedited run.

    .venv/Scripts/python exp4c_eval.py
"""
import json
from pathlib import Path

import librosa
import numpy as np

import meter

ROOT = Path("runs/exp4c")


def beat_bpm(audio):
    y = librosa.resample(meter.mono(audio), orig_sr=meter.SR, target_sr=22050)
    _, beats = librosa.beat.beat_track(y=y, sr=22050, units="time")
    return float(60 / np.median(np.diff(beats))) if len(beats) > 3 else float("nan")


def stats(audio, t0, t1):
    seg = meter.segment(audio, t0, t1)
    f = meter.features(seg)
    return dict(bpm=beat_bpm(seg), energy=meter.energy_score(f), lufs=f["lufs"],
                centroid=f["centroid_hz"], onset=f["onset_rate"])


def main():
    orig = meter.load_run(ROOT / "folk_orig")
    ref0 = Path("runs/exp1/folk_gen_W0/semantic.i32")
    same_as_exp1 = ref0.exists() and np.array_equal(np.fromfile(ref0, np.int32), orig.semantic)
    rows = []
    lines = [f"orig semantic tokens equal exp1 folk_gen_W0: {same_as_exp1}", "",
             "| edit | tokens | first diff (s) | bpm before | bpm after | d_energy after | d_lufs after | d_centroid after | d_onset after |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(ROOT.glob("folk_*")):
        r = meter.load_run(d)
        n = min(len(r.semantic), len(orig.semantic))
        neq = np.nonzero(r.semantic[:n] != orig.semantic[:n])[0]
        split = (neq[0] if len(neq) else n) / 25
        dur = r.audio.shape[1] / meter.SR
        dur0 = orig.audio.shape[1] / meter.SR
        before = stats(r.audio, 5, max(10, min(split, dur * 0.5)))
        after = stats(r.audio, dur * 0.65, dur)
        after0 = stats(orig.audio, dur0 * 0.65, dur0)
        row = dict(edit=d.name[5:], tokens=int(len(r.semantic)), split_s=split, before=before, after=after,
                   after_orig=after0)
        rows.append(row)
        lines.append(f"| {row['edit']} | {len(r.semantic)} | {split:.1f} | {before['bpm']:.0f} | {after['bpm']:.0f} | "
                     f"{after['energy'] - after0['energy']:+.2f} | {after['lufs'] - after0['lufs']:+.2f} | "
                     f"{after['centroid'] - after0['centroid']:+.0f} | {after['onset'] - after0['onset']:+.2f} |")
    t = "\n".join(lines)
    print(t)
    Path("tables/exp4c_plan_edits.md").write_text(t + "\n")
    Path("tables/exp4c_plan_edits.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
