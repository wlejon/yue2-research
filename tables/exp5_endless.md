# Exp 5: endless streams of 10 s chunks

Streams in `runs\exp5`; `exp5_eval.py`. Memory = tokens of history each chunk is opened by (25 per second). Every chunk: splice (semantic) + inpaint (NAR) + decode on one 4090, INT8 AR, 32 ODE steps.

## Cost per 10 s chunk (plain streams, mean)

| memory | splice s | inpaint s | decode s | GPU duty |
|---|---|---|---|---|
| 125 | 0.76 | 1.69 | 0.27 | 27% |
| 250 | 0.77 | 1.79 | 0.29 | 29% |
| 500 | 0.80 | 1.85 | 0.32 | 30% |
| 1000 | 0.86 | 1.74 | 0.29 | 29% |
| 2000 | 0.93 | 1.73 | 0.29 | 29% |

A bridge: the fresh clip 0.74 s + the transition (4 candidates, scored) 3.42 s of semantic sampling.
With CFG (two branches): splice 0.82 s per chunk.

## Seams and the beat

Seam z: flux at a join in units of the stream's own flux spread (0 = like anywhere else). Beat carry: onset strength on the beat grid extended across a point, over the mean (about 1 = random phase).

| group | streams | seam z | seam z max | joins > p95 | beat carry at joins | beat carry mid-chunk |
|---|---|---|---|---|---|---|
| const cfg 2 m500 | 1 | -0.06 | 1.95 | 0.00 | 0.91 | 1.28 |
| const cfg 3 m500 | 1 | 0.11 | 1.74 | 0.00 | 5.66 | 5.42 |
| const m1000 | 2 | -0.32 | 1.89 | 0.00 | 2.22 | 2.14 |
| const m125 | 2 | -0.12 | 3.80 | 0.06 | 3.17 | 3.18 |
| const m2000 | 2 | -0.36 | 2.25 | 0.03 | 2.29 | 2.35 |
| const m250 | 2 | -0.22 | 2.48 | 0.03 | 2.27 | 2.47 |
| const m500 | 2 | -0.36 | 2.57 | 0.03 | 3.10 | 2.78 |
| long | 1 | -0.02 | 3.27 | 0.06 | 4.93 | 5.40 |
| ref | 8 | -0.01 | 3.66 | 0.03 | 2.24 | 2.28 |
| score | 3 | 0.30 | 3.89 | 0.06 | 2.00 | 2.01 |
| score bridge | 2 | -0.15 | 1.53 | 0.00 | 1.55 | 2.24 |
| tour bridge cfg 2 m250 | 2 | 0.35 | 2.75 | 0.08 | 1.84 | 2.30 |
| tour bridge m250 | 2 | 0.13 | 3.17 | 0.08 | 2.39 | 3.32 |
| tour bridge m500 | 2 | 0.32 | 4.32 | 0.16 | 2.12 | 2.57 |
| tour cfg 1.5 m250 | 2 | 0.23 | 3.84 | 0.13 | 2.44 | 2.41 |
| tour cfg 1.5 m500 | 2 | -0.07 | 2.92 | 0.03 | 2.42 | 2.34 |
| tour cfg 2 m250 | 2 | 0.40 | 3.99 | 0.08 | 2.39 | 2.42 |
| tour cfg 2 m500 | 2 | 0.25 | 4.06 | 0.13 | 1.87 | 1.97 |
| tour cfg 3 m250 | 2 | -0.14 | 2.47 | 0.03 | 2.34 | 2.45 |
| tour cfg 3 m500 | 2 | 0.05 | 2.87 | 0.05 | 3.48 | 3.37 |
| tour m1000 | 2 | -0.13 | 2.23 | 0.03 | 2.65 | 3.11 |
| tour m125 | 2 | 0.12 | 2.92 | 0.05 | 1.51 | 1.53 |
| tour m2000 | 2 | -0.02 | 2.62 | 0.03 | 3.12 | 3.24 |
| tour m250 | 2 | -0.23 | 1.73 | 0.00 | 2.07 | 2.91 |
| tour m500 | 2 | -0.15 | 3.42 | 0.03 | 3.11 | 3.07 |
| voice | 2 | -0.07 | 3.39 | 0.03 | 1.23 | 1.03 |

## Constant styles: does a stream stay itself?

Reference streams (fresh starts, 2 seeds per style): self-similarity folk 0.852, synth 0.889, metal 0.848, lofi 0.860; their chunks' nearest centroid is their own style folk 100%, synth 100%, metal 91%, lofi 100%.

| stream | chunks | norm min..max | breaks | silent chunks (first) | CLAP consecutive | CLAP to ref first / last | chunks heard as another style | tempo off (median %, share within 4%) |
|---|---|---|---|---|---|---|---|---|
| const_folk_m1000 | 18 | 7.28..8.40 | 0 | 0 (-1) | 0.925 | 0.836 / 0.777 | 0 | 33.7, 0% |
| const_folk_m125 | 18 | 7.65..8.50 | 0 | 0 (-1) | 0.891 | 0.852 / 0.728 | 0 | 0.3, 88% |
| const_folk_m2000 | 18 | 7.32..9.02 | 0 | 4 (11) | 0.798 | 0.839 / 0.191 | 7 | 33.7, 6% |
| const_folk_m250 | 18 | 7.43..8.41 | 0 | 1 (5) | 0.869 | 0.685 / 0.831 | 4 | 33.7, 41% |
| const_folk_m500 | 18 | 7.74..8.40 | 0 | 0 (-1) | 0.890 | 0.854 / 0.792 | 0 | 0.3, 53% |
| const_folk_m500_cfg2 | 18 | 7.85..8.44 | 0 | 0 (-1) | 0.859 | 0.824 / 0.814 | 0 | 0.3, 100% |
| const_folk_m500_cfg3 | 18 | 7.82..8.27 | 0 | 0 (-1) | 0.893 | 0.827 / 0.885 | 0 | 4.0, 0% |
| const_synth_m1000 | 18 | 7.85..8.48 | 0 | 1 (10) | 0.834 | 0.883 / 0.689 | 2 | 32.4, 41% |
| const_synth_m125 | 18 | 7.77..8.35 | 0 | 0 (-1) | 0.924 | 0.868 / 0.869 | 0 | 4.3, 47% |
| const_synth_m2000 | 18 | 7.93..8.46 | 0 | 0 (-1) | 0.788 | 0.878 / 0.771 | 0 | 32.4, 47% |
| const_synth_m250 | 18 | 7.99..8.59 | 0 | 0 (-1) | 0.908 | 0.842 / 0.845 | 0 | 0.5, 65% |
| const_synth_m500 | 18 | 7.77..8.46 | 0 | 0 (-1) | 0.885 | 0.884 / 0.808 | 2 | 32.4, 24% |
| long_folk_m500 | 36 | 7.35..8.15 | 0 | 0 (-1) | 0.889 | 0.822 / 0.835 | 0 | 0.3, 86% |
| ref_folk_s11 | 12 | 7.47..8.40 | 0 | 0 (-1) | 0.853 | 0.875 / 0.853 | 0 | 4.0, 0% |
| ref_folk_s12 | 12 | 7.71..8.71 | 0 | 0 (-1) | 0.868 | 0.832 / 0.837 | 0 | 27.7, 0% |
| ref_lofi_s11 | 12 | 7.68..8.65 | 0 | 0 (-1) | 0.885 | 0.816 / 0.841 | 0 | 7.7, 0% |
| ref_lofi_s12 | 12 | 7.50..8.27 | 0 | 0 (-1) | 0.894 | 0.865 / 0.875 | 0 | 23.8, 0% |
| ref_metal_s11 | 12 | 7.11..8.22 | 0 | 0 (-1) | 0.807 | 0.778 / 0.755 | 2 | 2.5, 100% |
| ref_metal_s12 | 12 | 7.20..8.02 | 0 | 0 (-1) | 0.895 | 0.919 / 0.846 | 0 | 2.5, 91% |
| ref_synth_s11 | 12 | 7.85..8.25 | 0 | 0 (-1) | 0.861 | 0.887 / 0.874 | 0 | 36.9, 27% |
| ref_synth_s12 | 12 | 7.64..8.70 | 0 | 0 (-1) | 0.888 | 0.878 / 0.919 | 0 | 37.0, 18% |

## Mobility: after a style change, which style is heard

The tour: folk, synth, metal, lofi, folk, 4 chunks (40 s) each. Each change X -> Y: the share of Y's 4 chunks whose nearest reference is Y, and the chunk the first such arrives (0 = the first chunk after the change). Up to 8 changes per row (2 seeds); a change whose last chunk before it was already heard as Y is left out.

| condition | Y heard (share of chunks) | changes reached in 40 s | first arrival (median) | per change (hits) |
|---|---|---|---|---|
| bridge cfg 2 m250 | 75% | 100% | 1.0 | sy:.xxx me:.xxx lo:.xxx fo:.xxx sy:.xxx me:.xxx lo:.xxx fo:.xxx |
| bridge m250 | 75% | 100% | 1.0 | sy:.xxx me:.xxx lo:.xxx fo:.xxx sy:.xxx me:.xxx lo:.xxx fo:.xxx |
| bridge m500 | 69% | 100% | 1.0 | sy:.xxx me:.xxx lo:.xxx fo:.xxx sy:.xxx me:.xxx lo:.xxx fo:.x.. |
| plain cfg 1.5 m250 | 19% | 25% | 1.0 | sy:.... me:xxxx lo:.... fo:.... sy:..xx me:.... lo:.... fo:.... |
| plain cfg 1.5 m500 | 41% | 62% | 1.0 | sy:.xxx me:xxxx lo:.... fo:.xx. sy:.... me:..xx lo:.... fo:..xx |
| plain cfg 2 m250 | 21% | 43% | 2.0 | sy:...x me:.... lo:.... sy:..xx me:.xxx lo:.... fo:.... |
| plain cfg 2 m500 | 19% | 38% | 2.0 | sy:.... me:...x lo:.... fo:..xx sy:.... me:.xxx lo:.... fo:.... |
| plain cfg 3 m250 | 29% | 50% | 1.0 | sy:.xx. me:.xxx lo:.... sy:.... me:.xx. lo:.... |
| plain cfg 3 m500 | 32% | 57% | 1.0 | sy:.... me:xx.. lo:.... sy:..xx me:.x.x lo:.... fo:.xxx |
| plain m1000 | 0% | 0% | nan | sy:.... me:.... lo:.... fo:.... sy:.... me:.... lo:.... |
| plain m125 | 32% | 43% | 1.0 | sy:.... me:xxxx lo:.... sy:.... me:..xx lo:.... fo:.xxx |
| plain m2000 | 0% | 0% | nan | sy:.... me:.... lo:.... sy:.... me:.... lo:.... |
| plain m250 | 22% | 38% | 2.0 | sy:.... me:..xx lo:..xx fo:.... sy:.... me:.... lo:.xxx fo:.... |
| plain m500 | 7% | 29% | 0.5 | sy:.... me:.... lo:x... sy:.... me:.... lo:.x.. fo:.... |

## Every stream's nearest style per chunk

```
tour_m1000_s1                    fo fo fo fo fo fo fo fo fo fo fo sy sy sy sy sy sy sy sy sy
tour_m1000_s2                    fo fo fo fo fo fo fo fo fo fo sy sy fo fo fo fo fo fo fo fo
tour_m125_s1                     fo fo fo fo fo fo fo fo me me me me me sy me fo lo lo lo lo
tour_m125_s2                     fo fo fo fo fo fo fo fo fo fo me me me me sy sy sy fo fo fo
tour_m2000_s1                    fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo
tour_m2000_s2                    fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo fo
tour_m250_bridge_cfg2_s1         fo fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo fo fo
tour_m250_bridge_cfg2_s2         fo fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo fo fo
tour_m250_bridge_s1              fo fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo fo fo
tour_m250_bridge_s2              fo fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo fo fo
tour_m250_cfg1.5_s1              fo fo fo fo fo fo fo fo me me me me me sy sy sy sy sy sy sy
tour_m250_cfg1.5_s2              fo fo fo fo fo fo sy sy sy sy sy sy sy sy sy sy sy sy sy sy
tour_m250_cfg2_s1                fo fo fo fo fo fo fo sy sy sy sy sy sy sy fo fo fo fo fo fo
tour_m250_cfg2_s2                fo fo fo fo fo fo sy sy sy me me me me me me me me sy sy sy
tour_m250_cfg3_s1                fo fo fo fo fo sy sy fo fo me me me sy sy fo fo sy fo fo fo
tour_m250_cfg3_s2                fo fo fo fo fo fo lo lo sy me me fo fo fo fo fo fo fo fo fo
tour_m250_s1                     fo fo fo fo fo fo fo fo fo fo me me me me lo lo lo lo lo lo
tour_m250_s2                     fo fo fo fo fo fo fo fo fo fo fo fo fo lo lo lo lo lo lo lo
tour_m500_bridge_s1              fo fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo fo fo
tour_m500_bridge_s2              fo fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo lo lo
tour_m500_cfg1.5_s1              fo fo fo fo fo sy sy sy me me me me me me me me sy fo fo sy
tour_m500_cfg1.5_s2              fo fo fo fo fo fo fo fo fo fo me me me me me sy me me fo fo
tour_m500_cfg2_s1                fo fo fo fo fo fo fo fo fo fo fo me me me me me me lo fo fo
tour_m500_cfg2_s2                fo fo fo fo fo fo fo fo fo me me me me me me me me sy me sy
tour_m500_cfg3_s1                fo fo fo fo fo fo fo fo me me sy sy sy sy sy fo fo fo fo fo
tour_m500_cfg3_s2                fo fo fo fo fo fo sy sy sy me fo me sy fo fo sy sy fo fo fo
tour_m500_s1                     fo fo fo fo fo fo fo fo sy lo lo fo lo fo fo fo fo lo lo sy
tour_m500_s2                     fo fo fo fo fo fo fo fo fo fo fo fo fo lo sy sy sy sy lo sy
(asked)                          fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo fo fo fo
```

