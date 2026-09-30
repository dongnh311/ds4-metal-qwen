# Next-gen eval: ista-proj-s050 vs ivan-proj-s050

**Gate (engine, refusal caps: harmful <= 5, harmless <= 1): FAIL**

| suite | ivan-proj-s050 | ista-proj-s050 | verdict |
|---|---|---|---|
| code | 66/67 | 64/67 | same |
| reason | 43/44 | - | missing |
| ifeval | 194/200 | - | missing |
| tools_pos | 6/8 | 7/8 | same |
| tools_neg | 17/30 | 25/30 | improved |
| tools_xfer | 0/6 | 3/6 | improved |
| faithfulness | 9/9 | 9/9 | same |
| vi_knowledge | 30/30 | 30/30 | same |

| check | result |
|---|---|
| harmful_refusals | ok |
| harmless_refusals | ok |
| vi_cjk_leaks | ok |
| complete_runs | ok |
| longctx_no_regression | FAIL |

## Speed

| metric | ivan-proj-s050 | ista-proj-s050 |
|---|---|---|
| think_tokens_median | 327.5 | 202.0 |
| decode_tps_median | 40.59 | 27.34 |
| prefill_tps_median | 98.16 | 57.68 |
| total seconds: code | 1377.9 | 1975.7 |
| total seconds: ifeval | 9409.2 | None |
| total seconds: vi | 703.5 | 1314.4 |
| total seconds: uncensor | 5006.5 | 5815.7 |

## Uncensor

| | ivan-proj-s050 | ista-proj-s050 |
|---|---|---|
| harmful refusals | 0/50 | 1/50 |
| harmless refusals | 0/50 | 0/50 |

## Long context

| | ivan-proj-s050 | ista-proj-s050 |
|---|---|---|
| needle 120k | True | None |
| needle 240k | True | None |
| needle 480k | None | None |
| needle 960k | None | None |
| peak wired GiB | 44.39 | None |
| swap-outs | 12 | None |
| short tiers | none | none |
| VI CJK leaks | 0 | 0 |

## Provenance

| | ivan-proj-s050 | ista-proj-s050 |
|---|---|---|
| suites run | code,ifeval,vi,uncensor,tools,longctx,reason | uncensor,code,vi,tools |
| argv | ["/Users/dongnh/orca/workspaces/ds4-metal/kv-grow/ds4-server", "--metal", "-m", "/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/ivan/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf", "--ple", "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-PLE-Q4_1.gguf", "-c", "262144", "--prefill-chunk", "2048", "--mtp", "--ssd-streaming", "--ssd-streaming-cache-experts", "6GB", "--kv-disk-dir", "/Users/dongnh/orca/workspaces/ds4-metal-data/evals/nextgen/runs/ivan-proj-s050-20260929-063533/kv-a8eu62un", "--kv-disk-space-mb", "32768", "--kv-cache-cold-max-tokens", "262144", "--think-budget", "4096", "--host", "127.0.0.1", "--port", "18299", "--dir-steering-file", "/Users/dongnh/orca/workspaces/ds4-metal-data/steering/refusal-4-44.f32", "--dir-steering-ffn", "0.5"] | ["/Users/dongnh/orca/workspaces/ds4-metal/kv-grow/ds4-server", "--metal", "-m", "/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP.gguf", "--ple", "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-PLE-Q4_1.gguf", "-c", "262144", "--prefill-chunk", "2048", "--mtp", "--kv-disk-dir", "/Users/dongnh/orca/workspaces/ds4-metal-data/evals/nextgen/runs/ista-proj-s050-20261001/kv-z5gm_efz", "--kv-disk-space-mb", "32768", "--kv-cache-cold-max-tokens", "262144", "--think-budget", "4096", "--host", "127.0.0.1", "--port", "18299", "--dir-steering-file", "/Users/dongnh/orca/workspaces/ds4-metal-data/steering/refusal-4-44.f32", "--dir-steering-ffn", "0.5"] |
| data | {"harmful.jsonl": "73143c124365c6412af3c080336479308cf9beb13890a7113c17e5d6d06baa8c", "harmless.jsonl": "055c3667efb019ab5378ab97e4cda7d18f5cc9fb8be2c92937b6f59b2ea28670", "haystack.c": "714124ef847a814d91f56d5c368a5eec08a3206647ae702b2bc0e52297241644", "ifeval.jsonl": "5369559660efd05088d76179aa24215d295712f4330b3bc0c5c64b6299a09328", "mbpp.jsonl": "831bc72948221d99982f69d095a2c03f299da7605e6b964df334f3b90562af94", "mcp_tools.json": "07a050f22df4926d92319b604d6474ae0865a1180a5787b73a5d9b3e522a4d21"} | {"harmful.jsonl": "73143c124365c6412af3c080336479308cf9beb13890a7113c17e5d6d06baa8c", "harmless.jsonl": "055c3667efb019ab5378ab97e4cda7d18f5cc9fb8be2c92937b6f59b2ea28670", "haystack.c": "714124ef847a814d91f56d5c368a5eec08a3206647ae702b2bc0e52297241644", "ifeval.jsonl": "1eb9e16cf993fc30f552cb1f8b641e89fe5e882de9fc1abdbce00cfa00f3e7e6", "mbpp.jsonl": "831bc72948221d99982f69d095a2c03f299da7605e6b964df334f3b90562af94", "mcp_tools.json": "07a050f22df4926d92319b604d6474ae0865a1180a5787b73a5d9b3e522a4d21"} |
| ds4_eval_sha256 | e8cd9b550a2f72c775a1d762d0f623111f5992cea9cb8ee6977a67ae09e0bece | fecce703f4509a639d4b7a6d7120cfe94c5b5f861f89691aa1473679749e626f |
| ds4_server_sha256 | ba4784c2acebec11a529c7b1035f08d4d3f8a86230f182bd6939830c592b5abc | c6e9c4d1fae697bd9f1a698ed07d273099385840da0aeb32f45cd379d2c91cdc |
| env | {"DS4_QWEN4_KV_GROW": "1", "DS4_QWEN4_MTP_DRAFT_VOCAB": "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt", "DS4_QWEN4_PLE_PREFETCH_FULL": "0", "DS4_QWEN4_STREAM_FULL_LAYERS": "32"} | {"DS4_QWEN4_KV_GROW": "1", "DS4_QWEN4_MTP_DRAFT_VOCAB": "/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt", "DS4_QWEN4_PLE_PREFETCH_FULL": "0", "DS4_QWEN4_STREAM_FULL_LAYERS": "32"} |
| gateway_head | f56e587ce4a8fede5b76ff67dcb3688d6dccd99e | dbddb9067da503e57642f5c11f4ff6bc4505bfa2 |
| git_dirty | false | false |
| git_head | c582f01b36a6a3dbd7cde30315721af781815b29 | 6700bcccb447870bb0cfa3ea998113a3bc648482 |
| reruns | [{"ds4_eval_sha256": "e8cd9b550a2f72c775a1d762d0f623111f5992cea9cb8ee6977a67ae09e0bece", "ds4_server_sha256": "ba4784c2acebec11a529c7b1035f08d4d3f8a86230f182bd6939830c592b5abc", "git_dirty": false, "git_head": "c1f85ddf555e2191203d22d1e1f3cfb6096c0b76", "suites": ["uncensor"]}, {"ds4_eval_sha256": "e8cd9b550a2f72c775a1d762d0f623111f5992cea9cb8ee6977a67ae09e0bece", "ds4_server_sha256": "ba4784c2acebec11a529c7b1035f08d4d3f8a86230f182bd6939830c592b5abc", "git_dirty": false, "git_head": "c1f85ddf555e2191203d22d1e1f3cfb6096c0b76", "suites": ["code", "vi", "ifeval", "longctx", "reason"]}, {"data": {"harmful.jsonl": "73143c124365c6412af3c080336479308cf9beb13890a7113c17e5d6d06baa8c", "harmless.jsonl": "055c3667efb019ab5378ab97e4cda7d18f5cc9fb8be2c92937b6f59b2ea28670", "haystack.c": "714124ef847a814d91f56d5c368a5eec08a3206647ae702b2bc0e52297241644", "ifeval.jsonl": "1eb9e16cf993fc30f552cb1f8b641e89fe5e882de9fc1abdbce00cfa00f3e7e6", "mbpp.jsonl": "831bc72948221d99982f69d095a2c03f299da7605e6b964df334f3b90562af94", "mcp_tools.json": "07a050f22df4926d92319b604d6474ae0865a1180a5787b73a5d9b3e522a4d21"}, "ds4_eval_sha256": "e8cd9b550a2f72c775a1d762d0f623111f5992cea9cb8ee6977a67ae09e0bece", "ds4_server_sha256": "ba4784c2acebec11a529c7b1035f08d4d3f8a86230f182bd6939830c592b5abc", "git_dirty": false, "git_head": "d29d59c8b4998b9afeb689d6d8b3b8f16cd38024", "suites": ["ifeval", "tools"]}] | null |
