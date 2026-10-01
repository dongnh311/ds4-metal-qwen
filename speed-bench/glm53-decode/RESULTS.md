# GLM-5.3 decode program: measurements

Model: GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf, `--ssd-streaming --power 100 -c 262144`,
M5 Pro 64 GB, greedy, 256 tokens, prompt: the ISO-8601 duration task.

## Phase 0 (env only)

| Run | rep 1 t/s | rep 2 t/s | Output vs base |
|---|---|---|---|
| base | | | - |
| readahead off | | | |
| `--mtp` | | - | |

Selected-load time split (from timing-summary.txt):

## SP1 (decode gates)
