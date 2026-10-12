"""Write native job files (jobs/*.json) for the experiments.

    .venv/Scripts/python make_jobs.py exp1
"""
import json
import sys
from pathlib import Path

from prompts import SONGS

ROOT = Path(__file__).parent
JOBS = ROOT / "jobs"


def write(name, job):
    JOBS.mkdir(exist_ok=True)
    (JOBS / f"{name}.json").write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", JOBS / f"{name}.json", len(job["runs"]), "runs")


def exp1():
    """Streaming: protocol vs windowed generate (timing), and windowed synthesis
    of the protocol run's own tokens (seams)."""
    gen, syn = [], []
    for song, s in SONGS.items():
        base = dict(style=s["style"], lyrics=s["lyrics"], cot="full", seed=1001)
        gen.append(dict(name=f"{song}_gen_W0", **base, narChunkFrames=0))
        gen.append(dict(name=f"{song}_gen_W50", **base, narChunkFrames=50))
        for w in (0, 25, 50, 100, 250):
            syn.append(dict(name=f"{song}_syn_W{w}", mode="synthesize", **{"from": f"runs/exp1/{song}_gen_W0"},
                            narChunkFrames=w))
        # the natural spread: the same tokens, whole-song NAR, other noise
        syn.append(dict(name=f"{song}_syn_W0n2", mode="synthesize", **{"from": f"runs/exp1/{song}_gen_W0"},
                        narChunkFrames=0, seed=2))
    write("exp1_gen", dict(out="runs/exp1", arWeights="int8", vae=True, runs=gen))
    write("exp1_syn", dict(out="runs/exp1", arWeights="int8", vae=True, runs=syn))


CAPTURE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
NAR_EVALS = [0, 21, 42, 63]


def exp2():
    """Labelled corpus: genre x axis x pole x seed, with AR (semantic) and NAR
    residual means captured."""
    from prompts import GENRES, AXES, LYRICS
    runs = []
    for seed in (11, 12):
        for g, base in GENRES.items():
            runs.append(dict(name=f"{g}_neutral_s{seed}", style=base, lyrics=LYRICS["corpus"], cot="full", seed=seed,
                             capture=dict(layers=CAPTURE_LAYERS, plan=False, semantic=True, narEvals=NAR_EVALS)))
            for ax, poles in AXES.items():
                for pi, pole in enumerate(poles):
                    runs.append(dict(name=f"{g}_{ax}{'+-'[pi]}_s{seed}", style=f"{base}, {pole}",
                                     lyrics=LYRICS["corpus"], cot="full", seed=seed,
                                     capture=dict(layers=CAPTURE_LAYERS, plan=False, semantic=True,
                                                  narEvals=NAR_EVALS)))
    write("exp2_corpus", dict(out="corpus/exp2", arWeights="int8", vae=True, runs=runs))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
