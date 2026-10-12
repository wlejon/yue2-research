# yue2-research

**Can YuE2-3B be a radio station you can steer?**

YuE2-3B runs natively in brosoundml (`../brosoundml/include/brosoundml/yue2.h`),
bound to JS as `bro.music` (`../bro/docs/music-api.js`), and the YuE2 lab
(`../broworkshop/demos/yue2-lab`) drives every stage. The lab shows things,
but nothing in it is a control surface yet. The target use case is Jonny's:

> Continuous playback, like a radio station: a new song every couple of minutes,
> non-stop, with a small prefetch. Interrupt at any moment and ask for an
> "energetic" version, and have playback start near immediately, with more energy.

This repo answers the questions that decide whether that is buildable, and how,
**with measurements**. It keeps the posture of `../audio-research`: map the
mechanism before designing a use, and findings are measured numbers, not opinions.

1. **Streaming.** Can plan → semantic (AR) → synthesis (NAR flow matching) →
   VAE run as an overlapped pipeline on chunks? What window and context give
   seams indistinguishable from the whole-song NAR? What are time-to-first-audio
   and the steady-state realtime factor of every stage on a 4090?
2. **Where musical attributes live.** Energy, tempo, density, timbre, vocal
   presence: which stage, which layers, and the latent seam. Directions found by
   paired contrasts and probes, verified by steering and measuring the audio;
   how much steer each seam takes before the music breaks.
3. **Near-immediate interrupt.** Given semantic tokens already ahead of the
   playhead, which seam makes an audible "more energy" change soonest, and what
   is the request-to-audible latency of each?

## The stack under test

- **Native** (the timing oracle and the fast path): `bro-headless` with
  `bro.music`, from `../bro/build/Release`. Scripts in `native/`, driven by job
  files in `jobs/` (written by `make_jobs.py`). INT8 AR weights, 5 drafts.
- **Reference** (the introspection oracle): the `yue2` PyTorch package and
  weights in `../YuE2-3B` (CC BY-NC 4.0, used as data), through `lab.py`, for
  what the native seams cannot do yet: NAR windows with context and latent
  inpainting, NAR steering at arbitrary frames, AR continuation from an edited
  prompt.
- **Meters** (`meter.py`): integrated loudness (LUFS, pyloudnorm), dynamics,
  spectral centroid / flatness / rolloff, onset rate and strength, tempo,
  percussive share (HPSS), harmonic vocal-band share, low-end share; log-mel
  and spectral flux for seams.

## Layout

- `PLAN.md` — the experiments and their decision criteria.
- `FINDINGS.md` — results with numbers, and the recommended design.
- `native/` — the bro-headless harness (`gen.js`: generate / synthesize runs
  with steering, capture, and a callback timeline).
- `lab.py` — reference-model harness. `meter.py` — measurement kit.
- `seams.py` (Exp 1), `attributes.py` (Exp 2), `interrupt.py` (Exp 3) — analyses.
- `tables/`, `figures/` — committed results. `runs/`, `corpus/`, `cache/` —
  generated audio, latents and hidden states (git-ignored).

## Setup

    bash scripts/setup_venv.sh            # .venv over ../YuE2-3B/.venv + analysis packages
    .venv/Scripts/python make_jobs.py exp1
    YUE_JOB=jobs/exp1_gen.json ../bro/build/Release/bro-headless.exe native native/gen.js
    .venv/Scripts/python seams.py runs/exp1 --out tables/exp1_seams.md

Paths in a job are relative to the repo root (the working directory).
