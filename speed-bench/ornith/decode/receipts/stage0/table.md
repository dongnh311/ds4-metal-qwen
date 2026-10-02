| run | gen t/s | decode CBs | GPU busy | split ms/step (gdn-mix/gdn-moe/attn-mix/attn-moe/head) | verify split ms/chunk T=2 | spec stats | overlap |
|---|---|---|---|---|---|---|---|
| 128k-mtp-cb | 41.23 | 271 | 0.835 | missing | missing | Ornith spec stats: 128 cycles (1 plain), accept 0.787, 1.781 tokens/cycle, ms/cycle target 37.36 draft 4.57 host 0.11 outside 0.06, ms/token 23.63 | missing |
| 128k-mtp-split | 28.64 | 0 | missing | missing | 12.21/17.44/18.52/6.26/2.17 | missing | Ornith verify expert overlap: 3.31 of 8 per layer over 5120 layer-verifies |
| 128k-plain-cb | 34.06 | 270 | 0.978 | missing | missing | missing | missing |
| 128k-plain-split | 23.89 | 0 | missing | 10.86/9.17/16.50/3.35/2.18 | missing | missing | missing |
| 2k-mtp-cb | 84.02 | 392 | 0.927 | missing | missing | Ornith spec stats: 128 cycles (1 plain), accept 0.677, 1.672 tokens/cycle, ms/cycle target 18.23 draft 1.61 host 0.12 outside 0.06, ms/token 11.97 | missing |
| 2k-mtp-plain | 84.57 | 0 | missing | missing | missing | missing | missing |
| 2k-mtp-split | 40.39 | 0 | missing | missing | 11.41/16.39/4.39/5.44/2.20 | missing | Ornith verify expert overlap: 3.20 of 8 per layer over 5120 layer-verifies |
| 2k-mtp-t07 | 79.08 | 0 | missing | missing | missing | Ornith spec stats: 128 cycles (16 plain), accept 0.723, 1.633 tokens/cycle, ms/cycle target 17.78 draft 1.55 host 0.08 outside 0.99, ms/token 12.49 | missing |
| 2k-plain-cb | 70.01 | 397 | 0.974 | missing | missing | missing | missing |
| 2k-plain-l2 | 43.53 | 0 | missing | missing | missing | missing | missing |
| 2k-plain-split | 34.62 | 0 | missing | 10.73/9.13/3.99/2.99/2.22 | missing | missing | missing |
| 32k-mtp-cb | 64.93 | 379 | 0.932 | missing | missing | Ornith spec stats: 128 cycles (1 plain), accept 0.709, 1.703 tokens/cycle, ms/cycle target 24.08 draft 2.09 host 0.08 outside 0.04, ms/token 15.44 | missing |
| 32k-mtp-plain | 55.93 | 0 | missing | missing | missing | missing | missing |
| 32k-mtp-split | 34.64 | 0 | missing | missing | 11.69/16.75/10.65/5.88/2.17 | missing | Ornith verify expert overlap: 3.17 of 8 per layer over 5120 layer-verifies |
| 32k-mtp-t07 | 52.59 | 0 | missing | missing | missing | Ornith spec stats: 128 cycles (16 plain), accept 0.714, 1.625 tokens/cycle, ms/cycle target 26.76 draft 2.35 host 0.09 outside 1.13, ms/token 18.67 | missing |
| 32k-plain-cb | 52.45 | 371 | 0.956 | missing | missing | missing | missing |
| 32k-plain-split | 25.40 | 0 | missing | 12.33/10.72/9.87/3.73/2.65 | missing | missing | missing |
