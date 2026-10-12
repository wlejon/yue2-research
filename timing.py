"""Exp 1a/1d: stage timing and gapless-playback prebuffer from native run timelines.

For a streamed run, playback starts at the first audio chunk plus a prebuffer
P. Chunk k (arriving at t_k, starting at song second s_k) is in time when
t_k <= t_first + P + s_k, so the smallest gapless prebuffer is
max_k(t_k - t_first - s_k). Steady-state realtime factor = audio seconds made
per wall second after the first chunk.

    .venv/Scripts/python timing.py runs/exp1/*_gen_* [--out tables/exp1_timing.md]
"""
import argparse
import json
from pathlib import Path

SR = 48000


def summarize(d):
    info = json.loads((Path(d) / "run.json").read_text())
    tl, tm = info["timeline"], info["timing"]
    aud = tl.get("aud", [])
    row = dict(run=Path(d).name, W=info["options"].get("narChunkFrames", 0), cot=info["options"].get("cot", "full"),
               audio_s=info["frames"] / 25, wall_s=info["wall"], plan_s=tm["planSeconds"], plan_tok=tm["planTokens"],
               plan_tps=tm["planTps"], sem_s=tm["semanticSeconds"], sem_tps=tm["semanticTps"], nar_s=tm["synthSeconds"],
               nar_per_s=tm["narSecondsPerAudioSecond"], first_sem=tl.get("sem0"), first_lat=tl.get("lat0"),
               first_audio=tl.get("aud0"))
    if aud:
        t0 = aud[0][0]
        need = max(t - t0 - s0 / SR for t, s0, n in aud)
        row["prebuffer_s"] = max(0.0, need)
        end_audio = (aud[-1][1] + aud[-1][2]) / SR
        row["rtf_after_first"] = end_audio / max(1e-9, aud[-1][0] - t0)
        row["x_realtime"] = row["audio_s"] / row["wall_s"]
    return row


def table(rows, keys):
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[k]:.2f}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                                       for k in keys) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [summarize(d) for d in a.runs if (Path(d) / "run.json").exists()]
    keys = ["run", "W", "cot", "audio_s", "wall_s", "x_realtime", "plan_s", "plan_tok", "plan_tps", "sem_s", "sem_tps",
            "nar_s", "nar_per_s", "first_sem", "first_lat", "first_audio", "prebuffer_s", "rtf_after_first"]
    t = table(rows, keys)
    print(t)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(t + "\n")


if __name__ == "__main__":
    main()
