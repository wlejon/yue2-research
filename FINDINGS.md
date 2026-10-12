# YuE2 research: findings

Provenance:
- Native numbers come from the `bin/` snapshot (bro `87b92fb1`, brosoundml `ea4146a`, Release, INT8 AR) on GPU 0, a 4090 on bus 01.
- Reference numbers come from PyTorch `yue2` (BF16 eager, on bus 01 or GPU 1). Its timings are not latency claims; only the native timings are.
- Songs: folk, rock and synthpop, 170–184 s, cot full.
- The tables are in `tables/`. Generated audio and the corpus are gitignored.

## 1. Streaming: yes, with long enough windows

`narChunkFrames=50` changes the whole song, not just its seams:
- brightness (spectral centroid) drops to ×0.70;
- mel L1 against the whole-song render is 1.6–1.9, where re-rendering with only the noise changed differs by 0.32–0.36 (the natural spread).

Windows of 500 frames (20 s), with 25 frames of inpainted left context and 25 of lookahead, match whole-song quality:
- mel L1 0.27–0.34;
- no step at the seams (flux z ≈ 0);
- centroid ×0.95–0.97.

Starting with a 250-frame window and then using 500 is close behind (mel L1 0.31–0.46). More left context, the full song history or absolute positions don't rescue short windows; what matters is the size of each solved window.

Tables: `exp1_seams.md`, `exp1c_seams.md`.

## 2. A realtime radio is easy

Native stage speeds:

| stage | speed |
|---|---|
| plan | 2.4 s |
| semantic | 350 tok/s (14× realtime) |
| whole-song NAR | 0.09 s per audio second |
| VAE | 6.2 s per song |

About 36 s of GPU time builds a 170 s song, so prefetching one song is enough, at about 22% GPU duty.

Time to first audio with growing windows is about 4.3 s with cot full, or about 1.9 s without the plan.

Running two stages in separate processes on one GPU just time-slices: 28.4 s concurrent against 25.3 s back to back. Interleave the stages in one process instead.

Tables: `exp1_timing.md`, `latency_native.json`.

## 3. Attributes

- Energy, density, tempo-feel and brightness share one "arousal" direction at semantic layer 9 (cosine ≈ 0.95 between the axes). They separate by layer 21.
- Energy is cleanly linear:
  - semantic layers 21–24: d' 3.8, held-out-genre sign accuracy 1.0;
  - NAR: d' 2.5–3.8;
  - latent mean: d' 2.5.
- Tempo lives only in the plan's `Q:` header. Nothing changes it mid-song, a plan edit included: the beat stayed at 89 and 129 bpm.
- "Vocal / instrumental" has no audio effect when lyrics are present.
- Whole-song steering is clean in the calm direction down to −2 at every seam.
- In the energetic direction, spectral flatness rises 7–9× at +1 for semantic-layer steers (AR L9 and L21) and 3.4× for the latent offset. That may be a noisy wash; it needs listening.
- Past the cap:
  - AR steers at |scale| ≥ 3 never end the song (they hit 9000 tokens);
  - a latent offset of 4, and the NAR L21 steer at 4 or more, break the audio.

- Style axes at semantic layer 24 (whole-song steer, 4 held-out prompts, scales ±1–3; cosine with energy 0.41–0.56):
  - density: + is a full band (crest −6 dB at +1, 4/4 clips), − is sparse;
  - timbre: + is bright (centroid +1.3 kHz at +1, 4/4), − is warm and dark;
  - tempo feel: no scale raised the onset rate (0–1 of 4 at every positive scale). It acts like an energy steer, so tempo stays a plan (`Q:`) control.
  - Like energy, density and timbre are clean to −1, lose one clip in four at −2, and flatten the spectrum about 6× at +1. Layers 21 and 24 behave alike.
  - Shipped as stream knobs `density` and `timbre` (brosoundml, [−1.5, 1]); `export_dirs.py` writes them into `yue2_style_directions.inc`.

Tables: `exp2_prompt.md`, `exp2_seams.md`, `exp2c_steer_energy.md`, `exp2c_steer_{density,timbre,tempo}.md`.

## 4. Interrupt

Each run switches at 60 s and re-renders a 10 s span with the design window. An onset step at the switch above about 2 is audible.

| mechanism | native latency | effect at a safe scale | step at the switch |
|---|---|---|---|
| latent offset | ≈ 0.05 s | scale 0.5: +1.1–1.5 LUFS, +0.5 kHz; scale 1: +2.2–3.3 LUFS, +0.9–1.0 kHz | 0.02–0.21 |
| NAR L21 steer | ≈ 1–1.5 s | scale 2: energy +0.45–0.53 | 0.8–1.4 |
| AR L21 steer on the continuation | ≈ 2–2.5 s | scale 1: energy +0.42–0.58 on folk and rock; inconsistent on synthpop | below 1 |

- The latent offset is clean, but it only makes the song louder and brighter.
- The NAR L21 steer breaks at scale 4 or more (step 2.6–3.7). At 8–16 it adds 11–16 LUFS and clips.
- NAR L9 is a percussive knob: up to +3.2 onsets per second at scale 8.
- The AR L21 steer is a real arrangement change.
- Re-prompting the style, re-prompting the plan's tempo or labels, or re-rolling the continuation all do about nothing mid-song: the song's own history dominates.

- The style axes on a continuation (AR L24, scales −1.5, −1, 0.5, 1, folk and rock; `scripts/queue_style.sh`): every step at the switch is below 1 (−1.3 to +0.73, with the unsteered continuation at −1.3 to +0.18). Mid-song they do less than over a whole song, as every steer does:
  - density +1: centroid +0.4–0.8 kHz, flatness 2–2.5×;
  - timbre: inconsistent mid-song. Folk moved +0.8 kHz at 0.5 and not at 1; rock moved +0.6 kHz at 1.
  - tempo: as in exp2c, no onset-rate change.

Tables: `exp3_interrupt.md`, `exp3_{density,timbre,tempo}_interrupt.md`.

## 5. VRAM: why 14 GB, and about 5 GB to recover

Nothing in the current setup is silly. Peak for a default INT8 song:

| component | GB |
|---|---|
| CUDA context + Vulkan + engine | 1.50 |
| VAE | 0.50 |
| BF16 weights | 7.22 |
| INT8 copies, kept beside the BF16 | 1.35 |
| semantic KV, preallocated for prefix + 9000 tokens (112 KB per token) | ≈ 1.23 |
| NAR's copy of the whole song's AR K/V | ≈ 1.17 |
| workspace | ≈ 0.3 |
| **peak** | **13.30** |

CFG is off by default, so it isn't a second branch; with cfgScale 1.5 the peak is 14.46.

Savings:

| saving | GB | cost |
|---|---|---|
| drop the BF16 originals the INT8 copies shadow (prefill with INT8) | ≈ 2.7 | needs a parity check |
| NAR reads the session's K/V in place instead of copying it | ≈ 1.1 | none |
| 500-frame NAR windows | 1.1 (13.30 → 12.21) | none |
| grow the KV on demand | 0.5–0.75 | none |
| BF16 VAE | ≈ 0.25 | to be checked |

Together that is about 8 GB for a song.

Tables: `vram/*.json`.

## 6. Sliding AR context and plan streaming

A sliding AR context window isn't worth it. Scored by mean NLL under the full model:

| config | mean NLL | song ends? |
|---|---|---|
| full context | 3.52 | yes |
| last 2000 tokens kept | 3.47 | yes |
| last 1000 tokens | 3.71 | yes, but the last third drifts |
| last 500 tokens | 3.80 | yes, runs longer |
| last 250 tokens | 3.99 | yes, runs longer |
| re-based positions | 3.76–4.21 | never (6000-token cap) |

A whole song's KV is only 0.47 GB, so there is little to save anyway.

The plan can be built incrementally. Cut after four sections and then extended under the full lyrics, it keeps the same header and finishes every section. Its chords agree with an independent full plan at Jaccard 0.80–0.83.

The plan sits in the prefix ahead of every codec token, though, so editing it rewrites the song from 0.5 s on. Plan edits are a next-song control.

Tables: `exp4_slide_folk.json`, `exp4b_plan.json`, `exp4c_plan_edits.md`.

## Recommended design

**Streaming (brosoundml).** One `Yue2Stream` per song, in one process on one device:
- a plan (whole, or a few sections ahead);
- a live semantic session whose KV grows on demand;
- NAR windows of 250 frames, then 500, with 25 frames inpainted on the left and 25 of lookahead, reading the session's K/V in place;
- the existing streaming VAE.

**Radio.**
- Build the next song whole while the current one plays.
- The next song's prompt carries the listener's last steer, plus any tempo or key plan edits.
- Equal-power crossfade of 2–4 s at the outro.
- A skip starts a fresh stream (about 4.3 s to first audio).

**Interrupt, in two stages.**
1. Instantly: a latent offset of at most 1 with a 0.5 s ramp, re-decoded.
2. Within about 2.5 s: an AR L21-steered continuation from the playing frame + 1 s, inpainted and crossfaded in at a window edge. Then drop the offset.

The NAR L21 steer at ≤ 2 (or NAR L9 for percussion) is a texture knob.

**Lab controls:**
- an energy/brightness knob and a density/percussive knob;
- a plan panel for the next song;
- window and ODE-step controls (16 steps halve the NAR cost);
- the radio queue;
- meters with a break warning.

**Size:** about 3.9k LOC.

| area | part | LOC |
|---|---|---|
| brosoundml (≈ 2.3k) | inpainted windows + in-place K/V | 550 |
| | semantic continuation + KV growth | 400 |
| | stream scheduler | 450 |
| | section-ahead plan | 200 |
| | VRAM fixes | 250 |
| | bug guards | 80 |
| | JS binding (`src/api/native_soundml_music.cpp`) | 400 |
| bro (≈ 0.45k) | `docs/music-api.js` | 200 |
| | tests | 250 |
| yue2-lab (≈ 1.15k) | radio | 300 |
| | crossfade player | 250 |
| | interrupt UI | 250 |
| | plan panel | 200 |
| | meters | 150 |

## Bugs found (not fixed here)

- **B1. CUDA illegal memory access under a strong AR steer.** The error is `CUDA error 700 ... brotensor\src\cuda\tensor.cu:150 in cudaMemcpy D2H`, and the process dies.
  - Repro: job `soul_ar9-8_s21` in `jobs/exp2c.json`: style "soul, female vocal, Rhodes piano, bass", the corpus lyrics, cot full, seed 21, INT8, steer {layer 9, semantic only, scale −8, `cache/dirs/energy_semantic_L9_e-1.f32`}.
  - Likely cause: non-finite logits reach sampling and produce an out-of-range token index.
- **B2. An AR steer at |scale| ≥ 3 never ends the song.** It runs to 9000 tokens. A stream needs a cap tied to the plan's length, or an end bias.
- **B3. `require('./x')` in a `bro-headless <app> script.js` script resolves against the cwd**, not the script's directory, although `docs/brokit-api.js` says it resolves against the calling file.
