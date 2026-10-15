# Exp 9: lyrics rolled by the model's own pointer (head 14.10), end to end

Whisper large-v3 on the separated vocal (htdemucs_ft) per 10 s window, a line sung at 60% of its words.

## The FINDINGS 10 setup: 'voice', the 40-line sheet (40 lines), memory 500, 24 chunks

| stream | windows | words | distinct lines sung | furthest line | sung again later | words not in the sheet | sung lines in the chunk's window | |
|---|---|---|---|---|---|---|---|---|
| sheet_long | 23 | 224 | 12 | 24 | 24 | 3% | nan% | the whole sheet |
| roll_r1_w4 | 23 | 259 | 12 | 21 | 1 | 5% | 75% | 4 lines, 1 further each chunk |
| roll_r2_w4 | 23 | 129 | 15 | 39 | 1 | 24% | 88% | 4 lines, 2 further each chunk |
| roll_adapt_w4 | 23 | 254 | 24 | 24 | 4 | 17% | 87% | 4 lines after the last line Whisper heard on the mix (+ 3-chunk stall rule) |
| roll_adapt_w6 | 23 | 183 | 18 | 24 | 7 | 9% | 88% | 6 lines, the same |
| **head_w4** | 23 | 280 | 13 | 20 | 2 | 19% | 81% | **4 lines after the furthest line head 14.10 reached (no Whisper, no stall rule)** |

## The six exp8 voices (14 chunks, their sheets, the same seeds)

| stream | windows | words | distinct lines sung | furthest line | sung again later | words not in the sheet | sung lines in the chunk's window | |
|---|---|---|---|---|---|---|---|---|
| rock_m_harbor | 13 | 107 | 7 | 10 | 1 | 21% | 91% | 1 line further each chunk (exp8) |
| **rock_m_harbor** | 13 | 36 | 6 | 11 | 0 | 31% | 100% | **head-rolled** |
| folk_m_desert | 13 | 8 | 1 | 1 | 0 | 25% | 100% | 1 line further each chunk (exp8) |
| **folk_m_desert** | 13 | 17 | 2 | 5 | 0 | 29% | 100% | **head-rolled** |
| soul_f_harbor | 13 | 37 | 6 | 15 | 0 | 5% | 100% | 1 line further each chunk (exp8) |
| **soul_f_harbor** | 13 | 47 | 2 | 7 | 0 | 51% | 100% | **head-rolled** |
| country_m_desert | 13 | 58 | 10 | 15 | 1 | 14% | 100% | 1 line further each chunk (exp8) |
| **country_m_desert** | 13 | 36 | 5 | 11 | 0 | 17% | 100% | **head-rolled** |
| edm_f_desert | 13 | 37 | 4 | 11 | 0 | 35% | 100% | 1 line further each chunk (exp8) |
| **edm_f_desert** | 13 | 68 | 6 | 16 | 0 | 43% | 83% | **head-rolled** |
| ballad_f_harbor | 13 | 44 | 9 | 15 | 0 | 11% | 100% | 1 line further each chunk (exp8) |
| **ballad_f_harbor** | 13 | 79 | 6 | 8 | 0 | 48% | 100% | **head-rolled** |
| mean, 1 line a chunk | | | 6.2 | 11.2 | 0.3 | 19% | 98% | |
| mean, head-rolled | | | 4.5 | 9.7 | 0.0 | 36% | 97% | |

## Where the head moved each window (furthest line reached per chunk, -1 = none)

```
head_w4            -1 1 5 -1 -1 7 11 -1 15 -1 -1 -1 17 -1 -1 18 -1 -1 -1 -1 -1 21 23 25
rock_m_harbor      -1 0 2 4 5 9 -1 11 14 -1 -1 -1 -1 18
folk_m_desert      -1 -1 1 5 -1 -1 -1 -1 -1 -1 9 -1 -1 -1
soul_f_harbor      -1 1 -1 -1 2 -1 3 -1 7 -1 11 -1 -1 12
country_m_desert   -1 -1 1 4 -1 5 -1 7 9 12 15 -1 -1 -1
edm_f_desert       -1 0 2 -1 5 6 -1 7 -1 -1 9 12 16 -1
ballad_f_harbor    -1 0 3 -1 -1 -1 5 -1 -1 -1 6 8 -1 9
```

