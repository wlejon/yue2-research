# Yue2Stream oracle (brosoundml test_yue2_stream, 4090, INT8 AR)

The native stream (brosoundml `Yue2Stream`) is measured against this repo's design and its whole-song renders.

## Quality

Setup:
- Given semantic tokens from `runs/exp1/<song>_gen_W0`, the first 1500 frames.
- The stream renders them unpaced as the first song. That is the streaming case: windows of 250 then 500, L25, R25 inpainted.
- The reference is the whole-song render of the same tokens and seed (`Yue2::synthesize`).
- Run dirs are in `runs/stream_oracle`. Run `seams.py runs/stream_oracle --ref syn_W0`; the full table is `stream_oracle_seams.md`.

| song | lat_cos | mel_l1 | ref noise (syn_W0n2) | mel_l1 first 250 / 250-750 / 750-1500 | seam_flux_z | centroid_x |
|---|---|---|---|---|---|---|
| folk | 0.818 | 0.423 | 0.342 | 0.679 / 0.255 / 0.450 | 0.157 | 0.990 |
| rock | 0.881 | 0.321 | 0.330 | 0.438 / 0.206 / 0.359 | -0.067 | 1.026 |
| synthpop | 0.902 | 0.280 | 0.315 | 0.392 / 0.202 / 0.294 | -0.072 | 1.011 |

- Rock and synthpop are inside the whole render's seed-to-seed spread.
- Folk is not (0.423 against 0.342), but it beats this repo's W250-500 design run on folk (0.460).
- The error sits in the 250-frame windows: the first window, and the last of a truncated song. The 500-frame window is at about 0.2-0.26.
- No flux step at the seams; the centroid ratio is ≥ 0.99.

A song built ahead (every song after the first while one plays) is solved in one NAR chunk, so it *is* the whole render. Test D: 3789 frames in one chunk, latent cosine 1.0000. Only the first song and a skip's target pay the window cost.

## Latency (paced, a simulated player)

Time to first audio, cot full (tonight-awake):
- 6.7 s in total: the plan 4.5 s, then 2.2 s.
- This repo's 4.3 s assumed a 2.4 s plan.
- After the plan, it matches the 1.9 s with no plan here.

Interrupt +1, request to audible on the player's clock:

| stage | measured | budget |
|---|---|---|
| 1 (latent offset) | 0.75 s | the emit lead |
| 2 (AR L21 continuation, inpainted at a window edge) | 2.51 s | about 2.5 s |

- No click: the largest step within 10 ms of each landing is 0.5-0.8 of the program's p99 step.
- RMS rises after stage 1: 0.13 to 0.18.

Skip to a song that had not been built: 1.9-2.2 s of program until the cut-over.

## Device memory, ours (net of other processes)

| case | peak |
|---|---|
| weights (YuE2 INT8 AR, BF16 NAR, VAE decoder) | 6.4 GB |
| one song, live semantic, windows | 7.1 GB |
| playing + the next one building | 7.5 GB |
| a song built ahead in one 3789-frame chunk | 8.3 GB |

That is down from 13.3 GB here: the AR BF16 weights are dropped and dequantized just in time.

## Bugs

- **B1** (non-finite logits): every logits row is checked before sampling. A blown-up steer raises "non-finite logits", and the model stays usable.
- **B2** (a steered song never ends): there is an end bias past 1.15× the plan's length estimate and a hard cap at max(1.5E, E+20 s). An arrangement steer of 4 ended at 243 s against a 304 s cap; it ran to 9000 tokens before.
