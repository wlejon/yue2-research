# Exp 8: does YuE2 know which lyric word it is singing?

## Single heads, rolling windows (4-6 lines): 9614 sung frames

| layer.head | right word | within one word | right line | lag (frames) |
|---|---|---|---|---|
| 14.10 | 67% | 84% | 85% | -2 |
| 24.14 | 49% | 75% | 80% | -10 |
| 20.14 | 41% | 60% | 66% | -8 |
| 24.15 | 34% | 61% | 67% | -12 |
| 18.7 | 29% | 50% | 60% | -10 |
| 20.15 | 27% | 38% | 50% | -6 |
| 18.6 | 26% | 56% | 63% | -12 |
| 19.1 | 20% | 31% | 42% | -4 |
| 19.5 | 14% | 25% | 38% | -6 |
| 18.1 | 13% | 24% | 37% | -12 |
| 17.4 | 13% | 20% | 32% | 0 |
| 17.5 | 12% | 20% | 32% | -4 |

## Single heads, the whole 40-line sheet: 4527 sung frames

| layer.head | right word | within one word | right line | lag (frames) |
|---|---|---|---|---|
| 14.10 | 73% | 87% | 84% | -2 |
| 24.14 | 47% | 77% | 76% | -10 |
| 20.14 | 36% | 54% | 54% | -8 |
| 18.6 | 31% | 62% | 63% | -12 |
| 24.15 | 31% | 52% | 56% | -12 |
| 18.7 | 30% | 51% | 57% | -10 |
| 20.15 | 13% | 17% | 19% | -4 |
| 19.1 | 8% | 11% | 12% | -2 |
| 11.10 | 7% | 19% | 31% | -12 |
| 19.14 | 7% | 14% | 21% | -12 |
| 10.3 | 7% | 15% | 23% | -8 |
| 19.5 | 4% | 6% | 10% | -8 |

## A probe over every head's attention (leave one run out, rolling windows, lag -2)

9694 sung frames: right word 65%, within one word 82%, right line 83%

## Linear probes on the residual (leave one run out, rolling windows)

| after layer | singing or not | line of the window |
|---|---|---|
| 1 | 71% | 25% |
| 3 | 71% | 27% |
| 5 | 73% | 30% |
| 7 | 76% | 34% |
| 9 | 77% | 35% |
| 11 | 76% | 33% |
| 13 | 77% | 39% |
| 15 | 78% | 35% |
| 17 | 78% | 36% |
| 19 | 77% | 34% |
| 21 | 77% | 31% |
| 23 | 76% | 31% |
| 25 | 75% | 30% |
| 27 | 73% | 28% |

## The rolling window's question per chunk: the furthest window line sung (96 chunks)

| reader | right | one line off | missed a sung line | heard a line not sung |
|---|---|---|---|---|
| Whisper on the mix (the stream now) | 54% | 7% | 44% | 2% |
| head 14.10, lyric mass >= 0.05 | 80% | 9% | 0% | 20% |
| head 14.10, lyric mass >= 0.1 | 85% | 7% | 0% | 15% |
| head 14.10, lyric mass >= 0.2 | 89% | 4% | 0% | 11% |
| head 14.10, lyric mass >= 0.3 | 89% | 6% | 1% | 10% |
