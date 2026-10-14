# Exp 7: style changes that keep the beat; rolling lyrics

`exp7_eval.py`. Tours: memory 250 (10 s), 10 s chunks. Beat carry = onset strength on the beat grid of the 6 s before a join extended over the 4 s after it, over the mean (about 1 = random phase); a change's is the worst of its joins (the change chunk's, and a bridge's fresh clip's).

Scored references (folk score, one style each, 2 seeds): chunks heard as their own style synth 100%, metal 93%, jazz 100%, orch 100%.

| condition | streams | changes | Y heard (share of chunks) | reached in 40 s | beat carry at the change (median, min) | elsewhere |
|---|---|---|---|---|---|---|
| stour_m250 | 2 | 6 | 17% | 33% | 1.73, 0.75 | 2.17 |
| stour_m250_bridge | 7 | 19 | 42% | 84% | 1.54, 0.18 | 1.83 |
| stour_m250_bridge_cfg2 | 1 | 3 | 67% | 100% | 1.39, 0.70 | 1.50 |
| stour_m250_drop25 | 2 | 6 | 62% | 83% | 1.69, 0.67 | 1.38 |
| stour_m250_drop50 | 2 | 6 | 25% | 50% | 3.13, 0.53 | 1.99 |
| stour_m250_phase | 3 | 8 | 28% | 75% | 0.96, 0.12 | 2.35 |
| stour_m250_phase90 | 3 | 8 | 47% | 100% | 1.19, 0.17 | 1.86 |
| stour_m250_phaseaudio | 5 | 13 | 50% | 85% | 1.44, 0.15 | 2.12 |
| stour_m250_phasebridge | 5 | 13 | 42% | 85% | 1.27, 0.15 | 2.31 |
| stour_m250_phasekeep | 7 | 18 | 50% | 89% | 1.42, 0.18 | 2.30 |
| stour_m250_phasespec | 7 | 18 | 43% | 78% | 1.01, 0.21 | 1.82 |
| stour_m250_trim4 | 3 | 8 | 44% | 88% | 1.34, 0.28 | 2.25 |
| stour_m250_trim8 | 3 | 8 | 44% | 88% | 1.35, 0.28 | 2.25 |
| tour_m250 | 2 | 8 | 22% | 38% | 1.54, 1.24 | 2.06 |
| tour_m250_bridge | 2 | 8 | 75% | 100% | 1.42, 0.84 | 2.72 |
| tour_m250_drop12 | 2 | 8 | 78% | 100% | 1.55, 0.48 | 2.22 |
| tour_m250_drop25 | 2 | 7 | 57% | 71% | 1.29, 0.73 | 1.94 |
| tour_m250_drop50 | 2 | 8 | 31% | 50% | 3.64, 0.59 | 2.21 |

Scored tours, beat phase across joins (beats, 0 = the beat goes on, 0.5 = the off-beat; random is uniform 0..0.5, median 0.25):

| condition | beat period s | into the change chunk (median; each) | into the fresh clip (median; each; trims) | other joins (median; share < 0.1) |
|---|---|---|---|---|
| stour_m250 | 0.667 | 0.00; 0.03 0.00 0.00 0.00 0.00 0.02 | - | 0.00; 88% |
| stour_m250_bridge | 0.667 | 0.02; 0.02 0.02 0.02 0.00 0.00 0.08 0.00 0.08 0.00 0.00 0.02 0.22 0.00 0.00 0.00 0.00 0.00 0.02 0.02 0.19 0.02 | 0.31; 0.38 0.08 0.09 0.48 0.47 0.27 0.05 0.47 0.44 0.47 0.41 0.23 0.39 0.14 0.05 0.41 0.02 0.12 0.27 0.42 0.31; trims None None None None None None 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0.02; 62% |
| stour_m250_bridge_cfg2 | 0.667 | 0.02; 0.00 0.02 0.03 | 0.28; 0.20 0.44 0.28 | 0.00; 67% |
| stour_m250_drop25 | 0.667 | 0.06; 0.47 0.00 0.34 0.06 0.03 0.06 | - | 0.02; 75% |
| stour_m250_drop50 | 0.667 | 0.02; 0.05 0.02 0.00 0.05 0.03 0.02 | - | 0.02; 92% |
| stour_m250_phase | 0.667 | 0.00; 0.02 0.00 0.00 0.00 0.02 0.09 0.00 0.33 0.00 | 0.25; 0.22 0.38 0.50 0.25 0.17 0.31 0.00 0.34 0.08; trims 10 5 4 4 1 8 16 1 9 | 0.03; 67% |
| stour_m250_phase90 | 0.667 | 0.00; 0.02 0.00 0.00 0.00 0.02 0.03 0.00 0.34 0.00 | 0.05; 0.33 0.05 0.00 0.25 0.03 0.45 0.00 0.03 0.50; trims 12 3 2 4 3 8 16 7 8 | 0.02; 67% |
| stour_m250_phaseaudio | 0.667 | 0.00; 0.02 0.00 0.00 0.00 0.02 0.09 0.00 0.08 0.00 0.00 0.00 0.22 0.00 0.02 0.02 | 0.23; 0.22 0.39 0.50 0.02 0.05 0.34 0.11 0.06 0.50 0.23 0.09 0.42 0.02 0.39 0.41; trims 10 9 8 8 7 8 1 8 9 5 16 10 6 15 14 | 0.02; 64% |
| stour_m250_phasebridge | 0.667 | 0.02; 0.02 0.05 0.00 0.00 0.02 0.09 0.00 0.08 0.00 0.00 0.02 0.25 0.00 0.02 0.02 | 0.05; 0.48 0.00 0.03 0.02 0.03 0.47 0.11 0.00 0.48 0.47 0.08 0.05 0.02 0.39 0.02; trims 15 2 9 8 7 10 1 9 10 1 12 6 6 15 15 | 0.05; 60% |
| stour_m250_phasekeep | 0.667 | 0.00; 0.02 0.05 0.00 0.00 0.00 0.08 0.00 0.09 0.00 0.00 0.02 0.30 0.00 0.02 0.02 0.00 0.05 0.00 0.00 0.22 0.00 | 0.08; 0.50 0.00 0.03 0.02 0.03 0.47 0.11 0.00 0.50 0.47 0.08 0.05 0.02 0.39 0.02 0.06 0.11 0.02 0.34 0.42 0.20; trims 15 2 9 8 7 10 1 9 10 1 12 6 6 15 15 9 7 11 16 0 8 | 0.02; 65% |
| stour_m250_phasespec | 0.667 | 0.00; 0.02 0.00 0.00 0.00 0.00 0.08 0.00 0.09 0.00 0.00 0.00 0.22 0.00 0.00 0.48 0.00 0.05 0.00 0.00 0.19 0.00 | 0.05; 0.02 0.02 0.02 0.02 0.03 0.47 0.11 0.06 0.02 0.00 0.05 0.25 0.03 0.17 0.03 0.06 0.11 0.02 0.23 0.19 0.23; trims 7 5 12 8 7 10 1 10 3 9 3 11 7 7 15 9 7 11 1 16 7 | 0.02; 65% |
| stour_m250_trim4 | 0.667 | 0.02; 0.03 0.02 0.00 0.00 0.00 0.08 0.00 0.08 0.50 | 0.39; 0.39 0.12 0.16 0.48 0.47 0.27 0.05 0.50 0.45; trims 13 4 0 0 0 0 0 0 0 | 0.03; 63% |
| stour_m250_trim8 | 0.667 | 0.02; 0.03 0.02 0.00 0.00 0.02 0.08 0.00 0.08 0.50 | 0.38; 0.39 0.12 0.16 0.38 0.39 0.27 0.05 0.50 0.45; trims 13 4 0 2 0 0 0 0 0 | 0.03; 67% |

Per chunk, nearest style:

```
stour_m250_bridge_cfg2_s1    sy sy sy sy sy me me me me ja ja ja ja ja or or
stour_m250_bridge_s1         sy sy sy sy sy me or or or or or ja ja or or or
stour_m250_bridge_s2         sy sy sy sy sy me me me me or ja ja ja or or or
stour_m250_bridge_s3         sy sy sy sy sy or or me or ja ja ja ja me or ja
stour_m250_bridge_s4         sy sy sy sy sy sy sy sy sy or or or or me or or
stour_m250_bridge_s5         sy sy sy sy ja me me me me ja or or or or or or
stour_m250_bridge_s6         sy sy sy sy sy or or or or or or ja or me or or
stour_m250_bridge_s7         sy sy sy sy sy me or or or ja ja ja ja or or ja
stour_m250_drop25_s1         sy sy sy sy or or or or me or ja ja or or ja or
stour_m250_drop25_s2         sy sy sy sy me ja me me ja ja ja ja or or or sy
stour_m250_drop50_s1         sy sy sy sy sy or or or sy sy sy me sy ja or sy
stour_m250_drop50_s2         sy sy sy sy or sy sy sy sy ja ja ja or ja me or
stour_m250_phase90_s1        sy sy sy sy sy me or me me ja or ja ja or or or
stour_m250_phase90_s2        sy sy sy sy sy me or or or ja or or or ja or or
stour_m250_phase90_s3        sy sy sy sy sy or me or or ja ja ja ja me or or
stour_m250_phase_s1          sy sy sy sy sy me or or or me me me ja me or or
stour_m250_phase_s2          sy sy sy sy sy me or or or or or or or or or or
stour_m250_phase_s3          sy sy sy sy sy or me or me ja ja ja ja me or ja
stour_m250_phaseaudio_s1     sy sy sy sy sy me or or me or or or or or or or
stour_m250_phaseaudio_s2     sy sy sy sy sy me me me me ja ja or or ja or or
stour_m250_phaseaudio_s3     sy sy sy sy sy or or or ja ja ja ja ja me or ja
stour_m250_phaseaudio_s4     sy sy sy sy sy sy me me me or ja ja ja or or or
stour_m250_phaseaudio_s5     sy sy sy sy ja me ja me me ja ja ja ja or or or
stour_m250_phasebridge_s1    sy sy sy sy sy me or me me me or ja ja me or or
stour_m250_phasebridge_s2    sy sy sy sy sy me me me me or ja or or or or or
stour_m250_phasebridge_s3    sy sy sy sy sy or or or or ja ja ja ja me or me
stour_m250_phasebridge_s4    sy sy sy sy sy me sy sy or or or or ja or or or
stour_m250_phasebridge_s5    sy sy sy sy ja me ja me me ja ja ja ja or or or
stour_m250_phasekeep_s1      sy sy sy sy sy me or me me me or ja ja me or or
stour_m250_phasekeep_s2      sy sy sy sy sy me me me me ja ja ja ja or or or
stour_m250_phasekeep_s3      sy sy sy sy sy or or or or ja ja ja ja me or me
stour_m250_phasekeep_s4      sy sy sy sy sy me sy ja or or or or ja or or or
stour_m250_phasekeep_s5      sy sy sy sy ja me ja me ja ja ja ja ja or or or
stour_m250_phasekeep_s6      sy sy sy sy sy or me me me or or or or ja or ja
stour_m250_phasekeep_s7      sy sy sy sy sy me or or or sy ja ja ja or or or
stour_m250_phasespec_s1      sy sy sy sy sy me or or or me me me me or or or
stour_m250_phasespec_s2      sy sy sy sy sy me me me me ja ja ja ja or or or
stour_m250_phasespec_s3      sy sy sy sy sy or or or or ja ja ja ja me or or
stour_m250_phasespec_s4      sy sy sy sy sy sy me me me or or or or or or or
stour_m250_phasespec_s5      sy sy sy sy ja me me me me ja or or ja or or or
stour_m250_phasespec_s6      sy sy sy sy sy or me me me or or or or ja or ja
stour_m250_phasespec_s7      sy sy sy sy sy me or or me sy sy ja ja or or or
stour_m250_s1                sy sy sy sy sy sy sy or or ja ja ja ja ja ja ja
stour_m250_s2                sy sy sy sy sy sy sy sy sy sy or ja ja sy me me
stour_m250_trim4_s1          sy sy sy sy sy me me me me me or or or me or or
stour_m250_trim4_s2          sy sy sy sy sy me me me me or or ja ja ja or or
stour_m250_trim4_s3          sy sy sy sy sy or or me or ja ja ja ja me or ja
stour_m250_trim8_s1          sy sy sy sy sy me me me me me or or or me or or
stour_m250_trim8_s2          sy sy sy sy sy me me me me me or ja ja ja or or
stour_m250_trim8_s3          sy sy sy sy sy or or me or ja ja ja ja me or ja
tour_m250_bridge_s1          fo fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo fo fo
tour_m250_bridge_s2          fo fo fo fo fo sy sy sy sy me me me me lo lo lo lo fo fo fo
tour_m250_drop12_s1          fo fo fo fo sy sy sy sy sy sy me me lo lo lo lo fo fo fo fo
tour_m250_drop12_s2          fo fo fo fo lo lo lo sy fo fo me me lo lo lo lo fo fo fo fo
tour_m250_drop25_s1          fo fo fo fo sy sy sy sy sy sy me me sy sy fo fo sy fo fo fo
tour_m250_drop25_s2          fo fo fo fo fo fo fo fo me me me sy lo lo lo lo lo fo fo fo
tour_m250_drop50_s1          fo fo fo fo fo fo fo lo lo me me me sy sy sy sy fo fo fo lo
tour_m250_drop50_s2          fo fo fo fo fo fo lo lo lo lo me me lo lo sy sy sy sy sy sy
tour_m250_s1                 fo fo fo fo fo fo fo fo fo fo me me me me lo lo lo lo lo lo
tour_m250_s2                 fo fo fo fo fo fo fo fo fo fo fo fo fo lo lo lo lo lo lo lo
```

## Lyrics

Whisper large-v3 on the separated vocal per 10 s window. voice_m*: exp5's river sheet (16 distinct lines), the others the long sheet (40 lines, none repeated); roll_rN: N lines further each chunk, 4 shown.

| stream | windows | words | distinct lines sung | furthest line | sung again later | words not in the sheet | sung lines in the chunk's window |
|---|---|---|---|---|---|---|---|
| roll_adapt_w4 | 23 | 254 | 24 | 24 | 4 | 17% | 87% |
| roll_adapt_w6 | 23 | 183 | 18 | 24 | 7 | 9% | 88% |
| roll_r1_w4 | 23 | 259 | 12 | 21 | 1 | 5% | 75% |
| roll_r2_w4 | 23 | 129 | 15 | 39 | 1 | 24% | 88% |
| sheet_long | 23 | 224 | 12 | 24 | 24 | 3% | nan% |
| voice_m1000 | 17 | 38 | 5 | 13 | 0 | 0% | nan% |
| voice_m500 | 17 | 68 | 6 | 12 | 3 | 16% | nan% |

Lines sung per window (sheet line indices):

```
roll_adapt_w4  [] [0,1] [2,3] [4,5] [] [6] [7,8] [6,9] [4] [] [] [] [11,12] [13,14] [13,14] [] [] [] [10,16,17] [18,19] [20,21] [22,23] [20,21,24]
roll_adapt_w6  [] [0,1] [2,3,24] [0] [] [6] [] [10] [10] [6] [7,8] [9] [0] [] [13] [8,9] [10,11] [12,13] [14,15] [] [] [] [19,20,21]
roll_r1_w4     [] [1] [3] [4,5] [] [] [9] [10] [11] [12] [] [] [] [13] [10,13,16] [] [] [] [] [20,21] [20,21] [20,21] [20,21]
roll_r2_w4     [] [2,3,24] [4,5] [9] [] [] [14] [] [] [20,21] [23] [22,23] [] [] [] [] [32,35] [] [] [38,39] [] [38] []
sheet_long     [0] [] [] [4,5,6] [6,7,8] [9,10] [] [3,4] [] [8,9] [10,11] [2,3,24] [4,5,6] [7,8] [9,10] [2,11,24] [3,4] [6,7] [] [9,10,11] [2,11,24] [] []
voice_m1000    [] [] [3] [] [] [] [] [] [7,8] [] [] [] [] [] [] [] [6,13]
voice_m500     [] [] [3] [] [8] [] [10,11,12] [6] [] [10] [10,11,12] [] [] [] [] [] []
```

