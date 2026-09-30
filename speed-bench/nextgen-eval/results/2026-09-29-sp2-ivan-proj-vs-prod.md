# Next-gen eval: ivan-proj vs prod

**Gate: FAIL**

| suite | prod | ivan-proj | verdict |
|---|---|---|---|
| code | 65/67 | 64/67 | same |
| reason | 42/44 | 41/44 | same |
| ifeval | 60/60 | 58/60 | regressed |
| tools_pos | 8/8 | 8/8 | same |
| tools_neg | 2/6 | 1/6 | same |
| tools_xfer | 2/6 | 0/6 | regressed |
| faithfulness | 9/9 | 9/9 | same |
| vi_knowledge | 30/30 | 30/30 | same |

| check | result |
|---|---|
| harmful_refusals | ok |
| harmless_refusals | ok |
| vi_cjk_leaks | ok |
| needle_480k | FAIL |
| total_time_lower | ok |
| complete_runs | ok |
| longctx_no_regression | ok |

## Speed

| metric | prod | ivan-proj |
|---|---|---|
| think_tokens_median | 249.5 | 240 |
| decode_tps_median | 40.45 | 40.93 |
| prefill_tps_median | 94.88 | 114.87 |
| total seconds: code | 1045.5 | 1313.9 |
| total seconds: ifeval | 2456.7 | 3131.9 |
| total seconds: vi | 542.2 | 592.3 |
| total seconds: uncensor | 5891.5 | 4010.2 |

## Uncensor

| | prod | ivan-proj |
|---|---|---|
| harmful refusals | 1/50 | 0/50 |
| harmless refusals | 0/50 | 0/50 |

## Long context

| | prod | ivan-proj |
|---|---|---|
| needle 120k | True | True |
| needle 240k | True | True |
| needle 480k | None | None |
| needle 960k | None | None |
| peak wired GiB | 49.54 | 43.35 |
| swap-outs | 0 | 0 |
| VI CJK leaks | 0 | 0 |

## Provenance

| | prod | ivan-proj |
|---|---|---|
| suites run | code,ifeval,vi,uncensor,tools,longctx,reason | code,ifeval,vi,uncensor,tools,longctx,reason |
| argv | ["/Users/dongnh/orca/workspaces/ds4-metal/kv-grow/ds4-server", "--metal", "-m", "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf", "--ple", "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-PLE-Q4_1.gguf", "-c", "262144", "--prefill-chunk", "2048", "--mtp", "--ssd-streaming", "--ssd-streaming-cache-experts", "6GB", "--kv-disk-dir", "/Users/dongnh/orca/workspaces/ds4-metal-data/evals/nextgen/runs/prod-baseline-20260928/kv-1wznan3g", "--kv-disk-space-mb", "32768", "--kv-cache-cold-max-tokens", "262144", "--think-budget", "4096", "--host", "127.0.0.1", "--port", "18299"] | ["/Users/dongnh/orca/workspaces/ds4-metal/kv-grow/ds4-server", "--metal", "-m", "/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/ivan/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf", "--ple", "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-PLE-Q4_1.gguf", "-c", "262144", "--prefill-chunk", "2048", "--mtp", "--ssd-streaming", "--ssd-streaming-cache-experts", "6GB", "--kv-disk-dir", "/Users/dongnh/orca/workspaces/ds4-metal-data/evals/nextgen/runs/ivan-proj-20260929-012209/kv-qvz4jh6s", "--kv-disk-space-mb", "32768", "--kv-cache-cold-max-tokens", "262144", "--think-budget", "4096", "--host", "127.0.0.1", "--port", "18299", "--dir-steering-file", "/Users/dongnh/orca/workspaces/ds4-metal-data/steering/refusal-4-44.f32", "--dir-steering-ffn", "1"] |
| data | {"harmful.jsonl": "73143c124365c6412af3c080336479308cf9beb13890a7113c17e5d6d06baa8c", "harmless.jsonl": "055c3667efb019ab5378ab97e4cda7d18f5cc9fb8be2c92937b6f59b2ea28670", "haystack.c": "714124ef847a814d91f56d5c368a5eec08a3206647ae702b2bc0e52297241644", "ifeval.jsonl": "5369559660efd05088d76179aa24215d295712f4330b3bc0c5c64b6299a09328", "mbpp.jsonl": "831bc72948221d99982f69d095a2c03f299da7605e6b964df334f3b90562af94", "mcp_tools.json": "07a050f22df4926d92319b604d6474ae0865a1180a5787b73a5d9b3e522a4d21"} | {"harmful.jsonl": "73143c124365c6412af3c080336479308cf9beb13890a7113c17e5d6d06baa8c", "harmless.jsonl": "055c3667efb019ab5378ab97e4cda7d18f5cc9fb8be2c92937b6f59b2ea28670", "haystack.c": "714124ef847a814d91f56d5c368a5eec08a3206647ae702b2bc0e52297241644", "ifeval.jsonl": "5369559660efd05088d76179aa24215d295712f4330b3bc0c5c64b6299a09328", "mbpp.jsonl": "831bc72948221d99982f69d095a2c03f299da7605e6b964df334f3b90562af94", "mcp_tools.json": "07a050f22df4926d92319b604d6474ae0865a1180a5787b73a5d9b3e522a4d21"} |
| ds4_eval_sha256 | 71dfa147a9d95a2c2a8d474889deb06ecfd72f466f3e1d3965a1a3982dbeea56 | e8cd9b550a2f72c775a1d762d0f623111f5992cea9cb8ee6977a67ae09e0bece |
| ds4_server_sha256 | 252aaa109f7f75323e96b83a36c68f9815d21bf1d5d6ef5a21100df41172c79b | ba4784c2acebec11a529c7b1035f08d4d3f8a86230f182bd6939830c592b5abc |
| env | {"DS4_QWEN4_KV_GROW": "1", "DS4_QWEN4_MTP_DRAFT_VOCAB": "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt", "DS4_QWEN4_PLE_PREFETCH_FULL": "0", "DS4_QWEN4_STREAM_FULL_LAYERS": "32"} | {"DS4_QWEN4_KV_GROW": "1", "DS4_QWEN4_MTP_DRAFT_VOCAB": "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt", "DS4_QWEN4_PLE_PREFETCH_FULL": "0", "DS4_QWEN4_STREAM_FULL_LAYERS": "32"} |
| gateway_head | b2f00bae5207cf0cde1832af0e9bea181b44dc87 | f56e587ce4a8fede5b76ff67dcb3688d6dccd99e |
| git_dirty | false | false |
| git_head | a73ee1099ff11babf3f2640459e9123a1eb2fc84 | 8f043d11ca259ccb54ce980c92ef01de8d541c63 |
| reruns | [{"ds4_eval_sha256": "71dfa147a9d95a2c2a8d474889deb06ecfd72f466f3e1d3965a1a3982dbeea56", "ds4_server_sha256": "252aaa109f7f75323e96b83a36c68f9815d21bf1d5d6ef5a21100df41172c79b", "git_dirty": false, "git_head": "ba535ade318b5f8d5a83a434acd818152a3bdcac", "suites": ["reason"]}] | null |
