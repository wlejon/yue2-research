# Exp 6: one score, many styles

Renders in `runs\exp6` (`exp6.py`): the same ABC per song, cot full with the score supplied; vocals separated by htdemucs_ft.

## Tempo (beat-tracked) against the score's Q:

| render | tempo | Q: |
|---|---|---|
| folk_jazz_s1 | 89.1 | 90 |
| folk_jazz_s2 | 89.1 | 90 |
| folk_metal_s1 | 89.1 | 90 |
| folk_metal_s2 | 89.1 | 90 |
| folk_orch_s1 | 89.1 | 90 |
| folk_orch_s2 | 89.1 | 90 |
| folk_orig_s1 | 89.1 | 90 |
| folk_orig_s2 | 89.1 | 90 |
| folk_synth_s1 | 89.1 | 90 |
| folk_synth_s2 | 89.1 | 90 |
| rock_jazz_s1 | 129.2 | 65 |
| rock_jazz_s2 | 129.2 | 65 |
| rock_metal_s1 | 129.2 | 65 |
| rock_metal_s2 | 129.2 | 65 |
| rock_orch_s1 | 129.2 | 65 |
| rock_orch_s2 | 129.2 | 65 |
| rock_orig_s1 | 129.2 | 65 |
| rock_orig_s2 | 64.6 | 65 |
| rock_synth_s1 | 129.2 | 65 |
| rock_synth_s2 | 129.2 | 65 |

## Alignment between two renders of one score (vocal-stem DTW, seconds)

Pairs grouped: same style other seed, and other style. |drift| over sung frames; slope = the global tempo ratio of a linear fit; resid p90 = what one global stretch leaves.

| song | pairs | drift median | drift p90 | slope range | resid p90 median (max) |
|---|---|---|---|---|---|
| folk same style, other seed | 10 | 0.163 | 0.325 | 0.9976..1.0016 | 0.209 (0.411) |
| folk other style | 80 | 0.116 | 0.372 | 0.9985..1.0015 | 0.265 (0.665) |
| rock same style, other seed | 10 | 1.672 | 7.270 | 0.8954..1.0843 | 3.514 (6.089) |
| rock other style | 80 | 1.579 | 3.762 | 0.8765..1.1076 | 1.797 (8.066) |

## Alignment by windowed cross-correlation of the vocal stems' onset envelopes

Per 10 s window (5 s hop) of render i, the lag of render j within +-3 s that best matches its vocal onsets; windows with a correlation peak over 0.5 only (no DTW, so repeated choruses cannot be confused). within 50 ms = share of windows where j is within 50 ms of i; spread = p90 - p10 of the lag (a constant offset has spread 0 and one shift fixes it).

| song | pairs | windows kept | median abs lag | p90 abs lag | within 50 ms | lag spread median |
|---|---|---|---|---|---|---|
| folk same style, other seed | 10 | 91% | 0.140 | 0.160 | 19% | 0.020 |
| folk other style | 80 | 84% | 0.135 | 0.146 | 31% | 0.033 |
| rock same style, other seed | 10 | 58% | 1.652 | 2.377 | 20% | 0.322 |
| rock other style | 80 | 54% | 1.640 | 1.776 | 11% | 0.296 |

## Harmony: render i's vocal against render j's backing (chroma cosine over sung frames)

| condition | mean | min..max |
|---|---|---|
| matched (its own backing) | 0.526 | 0.422..0.592 |
| cross, DTW-warped (same score) | 0.525 | 0.402..0.634 |
| cross, unwarped (same score) | 0.518 | 0.396..0.620 |
| other song's backing (null) | 0.388 | 0.275..0.532 |

## Pairs

| i | j | drift med | drift p90 | slope | resid p90 | harmony warped / unwarped |
|---|---|---|---|---|---|---|
| folk_jazz_s1 | folk_jazz_s2 | 0.139 | 0.325 | 1.0005 | 0.192 | 0.471 / 0.472 |
| folk_jazz_s1 | folk_metal_s1 | 0.093 | 0.325 | 1.0004 | 0.239 | 0.451 / 0.463 |
| folk_jazz_s1 | folk_metal_s2 | 0.046 | 0.232 | 1.0006 | 0.243 | 0.456 / 0.465 |
| folk_jazz_s1 | folk_orch_s1 | 0.093 | 0.232 | 1.0002 | 0.203 | 0.474 / 0.467 |
| folk_jazz_s1 | folk_orch_s2 | 0.046 | 0.186 | 0.9998 | 0.171 | 0.402 / 0.396 |
| folk_jazz_s1 | folk_orig_s1 | 0.093 | 0.232 | 1.0002 | 0.241 | 0.513 / 0.472 |
| folk_jazz_s1 | folk_orig_s2 | 0.139 | 0.279 | 1.0005 | 0.176 | 0.508 / 0.504 |
| folk_jazz_s1 | folk_synth_s1 | 0.882 | 1.115 | 1.0007 | 0.244 | 0.445 / 0.425 |
| folk_jazz_s1 | folk_synth_s2 | 0.093 | 0.232 | 1.0000 | 0.228 | 0.453 / 0.448 |
| folk_jazz_s2 | folk_jazz_s1 | 0.186 | 0.325 | 0.9996 | 0.192 | 0.518 / 0.518 |
| folk_jazz_s2 | folk_metal_s1 | 0.232 | 0.557 | 0.9990 | 0.311 | 0.472 / 0.483 |
| folk_jazz_s2 | folk_metal_s2 | 0.139 | 0.418 | 0.9998 | 0.251 | 0.481 / 0.483 |
| folk_jazz_s2 | folk_orch_s1 | 0.232 | 0.418 | 0.9993 | 0.282 | 0.499 / 0.488 |
| folk_jazz_s2 | folk_orch_s2 | 0.186 | 0.372 | 0.9995 | 0.200 | 0.422 / 0.410 |
| folk_jazz_s2 | folk_orig_s1 | 0.186 | 0.464 | 0.9993 | 0.296 | 0.534 / 0.504 |
| folk_jazz_s2 | folk_orig_s2 | 0.046 | 0.232 | 0.9998 | 0.235 | 0.543 / 0.531 |
| folk_jazz_s2 | folk_synth_s1 | 0.743 | 0.882 | 1.0000 | 0.237 | 0.473 / 0.449 |
| folk_jazz_s2 | folk_synth_s2 | 0.186 | 0.372 | 0.9995 | 0.201 | 0.480 / 0.467 |
| folk_metal_s1 | folk_jazz_s1 | 0.093 | 0.372 | 1.0004 | 0.341 | 0.552 / 0.541 |
| folk_metal_s1 | folk_jazz_s2 | 0.232 | 0.650 | 1.0013 | 0.365 | 0.518 / 0.515 |
| folk_metal_s1 | folk_metal_s2 | 0.139 | 0.232 | 1.0008 | 0.159 | 0.505 / 0.503 |
| folk_metal_s1 | folk_orch_s1 | 0.093 | 0.218 | 1.0004 | 0.232 | 0.522 / 0.515 |
| folk_metal_s1 | folk_orch_s2 | 0.046 | 0.418 | 1.0005 | 0.379 | 0.431 / 0.426 |
| folk_metal_s1 | folk_orig_s1 | 0.093 | 0.576 | 1.0012 | 0.522 | 0.570 / 0.515 |
| folk_metal_s1 | folk_orig_s2 | 0.232 | 0.650 | 1.0014 | 0.401 | 0.560 / 0.546 |
| folk_metal_s1 | folk_synth_s1 | 0.975 | 1.533 | 0.9994 | 0.574 | 0.485 / 0.458 |
| folk_metal_s1 | folk_synth_s2 | 0.070 | 0.418 | 1.0006 | 0.319 | 0.497 / 0.487 |
| folk_metal_s2 | folk_jazz_s1 | 0.093 | 0.418 | 0.9995 | 0.416 | 0.558 / 0.551 |
| folk_metal_s2 | folk_jazz_s2 | 0.139 | 0.418 | 0.9997 | 0.323 | 0.529 / 0.526 |
| folk_metal_s2 | folk_metal_s1 | 0.139 | 0.279 | 0.9992 | 0.186 | 0.510 / 0.511 |
| folk_metal_s2 | folk_orch_s1 | 0.093 | 0.232 | 0.9994 | 0.215 | 0.527 / 0.526 |
| folk_metal_s2 | folk_orch_s2 | 0.093 | 0.279 | 0.9993 | 0.282 | 0.428 / 0.428 |
| folk_metal_s2 | folk_orig_s1 | 0.093 | 0.511 | 0.9997 | 0.500 | 0.561 / 0.522 |
| folk_metal_s2 | folk_orig_s2 | 0.093 | 0.464 | 0.9996 | 0.355 | 0.565 / 0.556 |
| folk_metal_s2 | folk_synth_s1 | 0.836 | 1.115 | 1.0000 | 0.400 | 0.491 / 0.472 |
| folk_metal_s2 | folk_synth_s2 | 0.139 | 0.279 | 0.9993 | 0.259 | 0.505 / 0.499 |
| folk_orch_s1 | folk_jazz_s1 | 0.093 | 0.232 | 1.0003 | 0.219 | 0.510 / 0.500 |
| folk_orch_s1 | folk_jazz_s2 | 0.186 | 0.418 | 1.0006 | 0.281 | 0.492 / 0.474 |
| folk_orch_s1 | folk_metal_s1 | 0.093 | 0.186 | 0.9997 | 0.212 | 0.469 / 0.470 |
| folk_orch_s1 | folk_metal_s2 | 0.093 | 0.209 | 1.0003 | 0.153 | 0.470 / 0.478 |
| folk_orch_s1 | folk_orch_s2 | 0.046 | 0.186 | 1.0006 | 0.221 | 0.408 / 0.412 |
| folk_orch_s1 | folk_orig_s1 | 0.046 | 0.279 | 1.0005 | 0.305 | 0.530 / 0.480 |
| folk_orch_s1 | folk_orig_s2 | 0.186 | 0.464 | 1.0004 | 0.279 | 0.532 / 0.514 |
| folk_orch_s1 | folk_synth_s1 | 0.929 | 1.161 | 1.0005 | 0.240 | 0.453 / 0.429 |
| folk_orch_s1 | folk_synth_s2 | 0.093 | 0.279 | 1.0002 | 0.244 | 0.463 / 0.471 |
| folk_orch_s2 | folk_jazz_s1 | 0.046 | 0.232 | 1.0005 | 0.205 | 0.514 / 0.496 |
| folk_orch_s2 | folk_jazz_s2 | 0.186 | 0.372 | 1.0008 | 0.206 | 0.475 / 0.463 |
| folk_orch_s2 | folk_metal_s1 | 0.046 | 0.325 | 1.0003 | 0.257 | 0.468 / 0.463 |
| folk_orch_s2 | folk_metal_s2 | 0.093 | 0.232 | 1.0008 | 0.197 | 0.478 / 0.473 |
| folk_orch_s2 | folk_orch_s1 | 0.046 | 0.186 | 1.0004 | 0.205 | 0.491 / 0.490 |
| folk_orch_s2 | folk_orig_s1 | 0.093 | 0.372 | 1.0002 | 0.370 | 0.518 / 0.472 |
| folk_orch_s2 | folk_orig_s2 | 0.186 | 0.367 | 1.0004 | 0.225 | 0.531 / 0.503 |
| folk_orch_s2 | folk_synth_s1 | 0.929 | 1.068 | 1.0010 | 0.240 | 0.458 / 0.422 |
| folk_orch_s2 | folk_synth_s2 | 0.070 | 0.186 | 1.0003 | 0.200 | 0.476 / 0.474 |
| folk_orig_s1 | folk_jazz_s1 | 0.093 | 0.255 | 1.0000 | 0.253 | 0.517 / 0.514 |
| folk_orig_s1 | folk_jazz_s2 | 0.186 | 0.464 | 1.0008 | 0.291 | 0.483 / 0.485 |
| folk_orig_s1 | folk_metal_s1 | 0.093 | 0.372 | 0.9994 | 0.354 | 0.470 / 0.474 |
| folk_orig_s1 | folk_metal_s2 | 0.093 | 0.325 | 1.0004 | 0.281 | 0.478 / 0.492 |
| folk_orig_s1 | folk_orch_s1 | 0.046 | 0.325 | 0.9997 | 0.307 | 0.479 / 0.475 |
| folk_orig_s1 | folk_orch_s2 | 0.093 | 0.464 | 1.0007 | 0.461 | 0.416 / 0.409 |
| folk_orig_s1 | folk_orig_s2 | 0.186 | 0.372 | 1.0004 | 0.213 | 0.520 / 0.512 |
| folk_orig_s1 | folk_synth_s1 | 0.929 | 1.115 | 1.0010 | 0.333 | 0.455 / 0.441 |
| folk_orig_s1 | folk_synth_s2 | 0.093 | 0.418 | 0.9999 | 0.413 | 0.458 / 0.456 |
| folk_orig_s2 | folk_jazz_s1 | 0.139 | 0.279 | 0.9995 | 0.178 | 0.534 / 0.535 |
| folk_orig_s2 | folk_jazz_s2 | 0.070 | 0.232 | 1.0002 | 0.239 | 0.504 / 0.514 |
| folk_orig_s2 | folk_metal_s1 | 0.232 | 0.511 | 0.9991 | 0.333 | 0.478 / 0.492 |
| folk_orig_s2 | folk_metal_s2 | 0.093 | 0.372 | 1.0000 | 0.245 | 0.490 / 0.499 |
| folk_orig_s2 | folk_orch_s1 | 0.186 | 0.464 | 0.9996 | 0.272 | 0.501 / 0.499 |
| folk_orig_s2 | folk_orch_s2 | 0.186 | 0.372 | 0.9997 | 0.216 | 0.428 / 0.418 |
| folk_orig_s2 | folk_orig_s1 | 0.186 | 0.372 | 0.9998 | 0.230 | 0.543 / 0.512 |
| folk_orig_s2 | folk_synth_s1 | 0.743 | 0.929 | 1.0005 | 0.304 | 0.474 / 0.457 |
| folk_orig_s2 | folk_synth_s2 | 0.186 | 0.325 | 0.9996 | 0.248 | 0.485 / 0.472 |
| folk_synth_s1 | folk_jazz_s1 | 0.882 | 1.115 | 0.9988 | 0.272 | 0.569 / 0.559 |
| folk_synth_s1 | folk_jazz_s2 | 0.743 | 0.882 | 1.0002 | 0.248 | 0.531 / 0.514 |
| folk_synth_s1 | folk_metal_s1 | 0.998 | 1.718 | 1.0015 | 0.665 | 0.485 / 0.500 |
| folk_synth_s1 | folk_metal_s2 | 0.882 | 1.207 | 0.9994 | 0.546 | 0.524 / 0.528 |
| folk_synth_s1 | folk_orch_s1 | 0.929 | 1.207 | 0.9989 | 0.357 | 0.537 / 0.522 |
| folk_synth_s1 | folk_orch_s2 | 0.929 | 1.073 | 0.9985 | 0.283 | 0.448 / 0.447 |
| folk_synth_s1 | folk_orig_s1 | 0.929 | 1.161 | 0.9990 | 0.404 | 0.578 / 0.530 |
| folk_synth_s1 | folk_orig_s2 | 0.743 | 0.929 | 0.9987 | 0.344 | 0.582 / 0.575 |
| folk_synth_s1 | folk_synth_s2 | 0.929 | 1.161 | 0.9976 | 0.411 | 0.510 / 0.505 |
| folk_synth_s2 | folk_jazz_s1 | 0.093 | 0.232 | 0.9999 | 0.258 | 0.547 / 0.536 |
| folk_synth_s2 | folk_jazz_s2 | 0.186 | 0.372 | 1.0008 | 0.222 | 0.519 / 0.510 |
| folk_synth_s2 | folk_metal_s1 | 0.093 | 0.464 | 1.0000 | 0.370 | 0.497 / 0.492 |
| folk_synth_s2 | folk_metal_s2 | 0.139 | 0.279 | 1.0008 | 0.241 | 0.516 / 0.512 |
| folk_synth_s2 | folk_orch_s1 | 0.093 | 0.325 | 1.0000 | 0.305 | 0.527 / 0.527 |
| folk_synth_s2 | folk_orch_s2 | 0.070 | 0.232 | 0.9998 | 0.233 | 0.435 / 0.434 |
| folk_synth_s2 | folk_orig_s1 | 0.093 | 0.418 | 1.0008 | 0.417 | 0.554 / 0.519 |
| folk_synth_s2 | folk_orig_s2 | 0.186 | 0.372 | 1.0003 | 0.252 | 0.557 / 0.540 |
| folk_synth_s2 | folk_synth_s1 | 0.929 | 1.115 | 1.0016 | 0.310 | 0.493 / 0.461 |
| rock_jazz_s1 | rock_jazz_s2 | 1.672 | 10.774 | 0.9228 | 6.089 | 0.604 / 0.569 |
| rock_jazz_s1 | rock_metal_s1 | 0.139 | 15.047 | 0.8986 | 6.386 | 0.541 / 0.544 |
| rock_jazz_s1 | rock_metal_s2 | 2.694 | 10.031 | 0.9324 | 8.013 | 0.542 / 0.551 |
| rock_jazz_s1 | rock_orch_s1 | 0.511 | 4.737 | 0.9756 | 3.212 | 0.544 / 0.542 |
| rock_jazz_s1 | rock_orch_s2 | 2.694 | 3.390 | 1.0171 | 1.534 | 0.539 / 0.489 |
| rock_jazz_s1 | rock_orig_s1 | 2.136 | 10.495 | 0.8792 | 5.196 | 0.553 / 0.560 |
| rock_jazz_s1 | rock_orig_s2 | 3.437 | 4.040 | 1.0092 | 0.611 | 0.534 / 0.508 |
| rock_jazz_s1 | rock_synth_s1 | 0.139 | 3.762 | 1.0183 | 3.096 | 0.515 / 0.513 |
| rock_jazz_s1 | rock_synth_s2 | 0.186 | 12.399 | 0.9057 | 5.565 | 0.553 / 0.553 |
| rock_jazz_s2 | rock_jazz_s1 | 1.672 | 7.839 | 0.9324 | 4.911 | 0.541 / 0.544 |
| rock_jazz_s2 | rock_metal_s1 | 1.579 | 7.793 | 0.9247 | 4.765 | 0.495 / 0.509 |
| rock_jazz_s2 | rock_metal_s2 | 1.022 | 5.378 | 0.9872 | 2.808 | 0.524 / 0.518 |
| rock_jazz_s2 | rock_orch_s1 | 1.254 | 7.282 | 0.9327 | 4.543 | 0.512 / 0.517 |
| rock_jazz_s2 | rock_orch_s2 | 1.068 | 1.393 | 1.0141 | 0.552 | 0.504 / 0.491 |
| rock_jazz_s2 | rock_orig_s1 | 0.511 | 1.440 | 1.0255 | 1.606 | 0.529 / 0.528 |
| rock_jazz_s2 | rock_orig_s2 | 1.765 | 3.474 | 1.0484 | 1.786 | 0.507 / 0.512 |
| rock_jazz_s2 | rock_synth_s1 | 1.579 | 2.313 | 1.0284 | 2.577 | 0.467 / 0.492 |
| rock_jazz_s2 | rock_synth_s2 | 1.579 | 1.811 | 0.9967 | 0.933 | 0.520 / 0.533 |
| rock_metal_s1 | rock_jazz_s1 | 0.093 | 0.464 | 1.0190 | 1.253 | 0.594 / 0.598 |
| rock_metal_s1 | rock_jazz_s2 | 1.579 | 1.811 | 1.0175 | 1.263 | 0.600 / 0.587 |
| rock_metal_s1 | rock_metal_s2 | 2.554 | 3.019 | 1.0439 | 1.760 | 0.556 / 0.568 |
| rock_metal_s1 | rock_orch_s1 | 0.372 | 0.604 | 1.0039 | 0.269 | 0.550 / 0.560 |
| rock_metal_s1 | rock_orch_s2 | 2.647 | 2.926 | 1.0204 | 1.285 | 0.520 / 0.492 |
| rock_metal_s1 | rock_orig_s1 | 2.043 | 2.229 | 0.9980 | 0.306 | 0.562 / 0.572 |
| rock_metal_s1 | rock_orig_s2 | 3.204 | 3.901 | 1.0541 | 2.545 | 0.526 / 0.504 |
| rock_metal_s1 | rock_synth_s1 | 0.093 | 1.393 | 1.0617 | 3.683 | 0.507 / 0.511 |
| rock_metal_s1 | rock_synth_s2 | 0.093 | 0.464 | 1.0009 | 0.423 | 0.545 / 0.557 |
| rock_metal_s2 | rock_jazz_s1 | 2.694 | 4.272 | 1.0543 | 3.784 | 0.603 / 0.600 |
| rock_metal_s2 | rock_jazz_s2 | 1.022 | 6.920 | 0.9461 | 2.633 | 0.634 / 0.613 |
| rock_metal_s2 | rock_metal_s1 | 2.554 | 6.701 | 0.9532 | 2.392 | 0.575 / 0.563 |
| rock_metal_s2 | rock_orch_s1 | 2.136 | 8.313 | 0.9485 | 2.797 | 0.569 / 0.573 |
| rock_metal_s2 | rock_orch_s2 | 0.186 | 0.604 | 0.9943 | 0.321 | 0.541 / 0.524 |
| rock_metal_s2 | rock_orig_s1 | 0.511 | 4.937 | 0.9506 | 2.491 | 0.579 / 0.584 |
| rock_metal_s2 | rock_orig_s2 | 0.836 | 12.399 | 1.0843 | 6.028 | 0.543 / 0.538 |
| rock_metal_s2 | rock_synth_s1 | 2.508 | 2.879 | 0.9924 | 0.323 | 0.515 / 0.529 |
| rock_metal_s2 | rock_synth_s2 | 2.554 | 9.209 | 0.9448 | 3.134 | 0.558 / 0.571 |
| rock_orch_s1 | rock_jazz_s1 | 0.464 | 0.929 | 1.0053 | 0.995 | 0.583 / 0.583 |
| rock_orch_s1 | rock_jazz_s2 | 1.161 | 1.625 | 1.0094 | 1.291 | 0.601 / 0.584 |
| rock_orch_s1 | rock_metal_s1 | 0.372 | 0.771 | 1.0109 | 1.038 | 0.557 / 0.553 |
| rock_orch_s1 | rock_metal_s2 | 2.136 | 4.950 | 1.0347 | 2.632 | 0.551 / 0.557 |
| rock_orch_s1 | rock_orch_s2 | 2.276 | 2.601 | 1.0217 | 2.073 | 0.541 / 0.503 |
| rock_orch_s1 | rock_orig_s1 | 1.672 | 1.997 | 0.9996 | 0.995 | 0.567 / 0.570 |
| rock_orch_s1 | rock_orig_s2 | 2.786 | 3.762 | 0.9950 | 2.159 | 0.539 / 0.513 |
| rock_orch_s1 | rock_synth_s1 | 0.418 | 8.313 | 1.0384 | 6.292 | 0.527 / 0.517 |
| rock_orch_s1 | rock_synth_s2 | 0.418 | 0.827 | 0.9948 | 0.384 | 0.546 / 0.550 |
| rock_orch_s2 | rock_jazz_s1 | 2.694 | 2.926 | 0.9982 | 0.312 | 0.590 / 0.589 |
| rock_orch_s2 | rock_jazz_s2 | 1.115 | 7.690 | 0.9456 | 3.200 | 0.602 / 0.586 |
| rock_orch_s2 | rock_metal_s1 | 2.647 | 8.805 | 0.9504 | 3.059 | 0.557 / 0.554 |
| rock_orch_s2 | rock_metal_s2 | 0.186 | 0.604 | 1.0057 | 0.298 | 0.547 / 0.551 |
| rock_orch_s2 | rock_orch_s1 | 2.322 | 8.248 | 0.9541 | 3.059 | 0.557 / 0.559 |
| rock_orch_s2 | rock_orig_s1 | 0.604 | 7.180 | 0.9474 | 3.346 | 0.561 / 0.559 |
| rock_orch_s2 | rock_orig_s2 | 0.743 | 1.300 | 1.0123 | 0.431 | 0.529 / 0.532 |
| rock_orch_s2 | rock_synth_s1 | 2.647 | 2.786 | 0.9986 | 0.327 | 0.494 / 0.513 |
| rock_orch_s2 | rock_synth_s2 | 2.694 | 11.081 | 0.9433 | 4.547 | 0.527 / 0.559 |
| rock_orig_s1 | rock_jazz_s1 | 2.136 | 10.495 | 1.1076 | 6.113 | 0.586 / 0.607 |
| rock_orig_s1 | rock_jazz_s2 | 0.511 | 0.743 | 0.9959 | 0.441 | 0.607 / 0.602 |
| rock_orig_s1 | rock_metal_s1 | 2.043 | 2.229 | 1.0124 | 0.726 | 0.574 / 0.559 |
| rock_orig_s1 | rock_metal_s2 | 0.511 | 0.975 | 1.0401 | 1.809 | 0.583 / 0.568 |
| rock_orig_s1 | rock_orch_s1 | 1.672 | 1.997 | 1.0126 | 0.682 | 0.572 / 0.577 |
| rock_orig_s1 | rock_orch_s2 | 0.557 | 4.398 | 1.0609 | 3.800 | 0.541 / 0.522 |
| rock_orig_s1 | rock_orig_s2 | 1.207 | 2.276 | 1.0843 | 4.324 | 0.533 / 0.545 |
| rock_orig_s1 | rock_synth_s1 | 2.090 | 9.260 | 1.0699 | 5.703 | 0.513 / 0.529 |
| rock_orig_s1 | rock_synth_s2 | 2.090 | 2.276 | 1.0019 | 0.483 | 0.554 / 0.572 |
| rock_orig_s2 | rock_jazz_s1 | 3.529 | 4.087 | 0.9891 | 0.441 | 0.576 / 0.568 |
| rock_orig_s2 | rock_jazz_s2 | 2.090 | 17.369 | 0.8834 | 6.306 | 0.584 / 0.570 |
| rock_orig_s2 | rock_metal_s1 | 3.483 | 19.087 | 0.8765 | 6.113 | 0.537 / 0.522 |
| rock_orig_s2 | rock_metal_s2 | 0.882 | 12.492 | 0.9003 | 5.138 | 0.544 / 0.534 |
| rock_orig_s2 | rock_orch_s1 | 2.926 | 6.734 | 1.0027 | 8.066 | 0.539 / 0.528 |
| rock_orig_s2 | rock_orch_s2 | 0.882 | 3.924 | 1.0138 | 3.185 | 0.516 / 0.493 |
| rock_orig_s2 | rock_orig_s1 | 1.440 | 14.489 | 0.8954 | 5.256 | 0.552 / 0.545 |
| rock_orig_s2 | rock_synth_s1 | 3.483 | 4.110 | 0.9814 | 0.580 | 0.488 / 0.494 |
| rock_orig_s2 | rock_synth_s2 | 3.529 | 16.115 | 0.8907 | 5.053 | 0.543 / 0.542 |
| rock_synth_s1 | rock_jazz_s1 | 0.093 | 0.464 | 0.9865 | 0.930 | 0.585 / 0.594 |
| rock_synth_s1 | rock_jazz_s2 | 1.625 | 9.316 | 0.9259 | 4.079 | 0.612 / 0.591 |
| rock_synth_s1 | rock_metal_s1 | 0.139 | 8.824 | 0.9257 | 4.361 | 0.555 / 0.554 |
| rock_synth_s1 | rock_metal_s2 | 2.508 | 2.926 | 0.9805 | 1.996 | 0.564 / 0.558 |
| rock_synth_s1 | rock_orch_s1 | 0.418 | 8.851 | 0.9250 | 4.356 | 0.538 / 0.563 |
| rock_synth_s1 | rock_orch_s2 | 2.647 | 2.833 | 1.0001 | 0.337 | 0.538 / 0.501 |
| rock_synth_s1 | rock_orig_s1 | 2.090 | 5.322 | 0.9442 | 3.738 | 0.564 / 0.569 |
| rock_synth_s1 | rock_orig_s2 | 3.251 | 3.901 | 1.0143 | 0.372 | 0.545 / 0.515 |
| rock_synth_s1 | rock_synth_s2 | 0.139 | 8.480 | 0.9313 | 3.969 | 0.570 / 0.566 |
| rock_synth_s2 | rock_jazz_s1 | 0.093 | 0.697 | 1.0212 | 1.490 | 0.607 / 0.614 |
| rock_synth_s2 | rock_jazz_s2 | 1.579 | 1.811 | 0.9961 | 0.358 | 0.627 / 0.620 |
| rock_synth_s2 | rock_metal_s1 | 0.093 | 0.464 | 1.0002 | 0.423 | 0.565 / 0.563 |
| rock_synth_s2 | rock_metal_s2 | 2.508 | 3.065 | 1.0302 | 1.273 | 0.553 / 0.573 |
| rock_synth_s2 | rock_orch_s1 | 0.372 | 0.604 | 1.0040 | 0.356 | 0.566 / 0.571 |
| rock_synth_s2 | rock_orch_s2 | 2.694 | 2.902 | 1.0207 | 1.239 | 0.525 / 0.505 |
| rock_synth_s2 | rock_orig_s1 | 2.090 | 2.183 | 0.9991 | 0.397 | 0.582 / 0.584 |
| rock_synth_s2 | rock_orig_s2 | 3.111 | 3.901 | 1.0373 | 1.377 | 0.527 / 0.514 |
| rock_synth_s2 | rock_synth_s1 | 0.093 | 0.418 | 1.0186 | 1.219 | 0.502 / 0.517 |
