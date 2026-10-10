# Pre-registration: Ornith native vision vs the describe sidecar (2026-10-10)

Written and committed before either arm ran, and before any model output on these images was read. The only model
outputs seen before this file were the E2E checks C1 and C2: the newspaper's paper, date and headline, and the code
screenshot's function and literal.

## Question

Does Ornith reading the pixels itself recover at least as many of the image's facts as Ornith reading the
gateway sidecar's description (Qwen3-VL-4B-Instruct-4bit)?

## Arms

Both arms use the same scratch ds4-server: the production Ornith-512K argv plus `--vision`. Settings: temperature 0,
`max_tokens` 600, thinking off. The question for every image is "Describe this image in detail, including any text
you can read."
- **native:** the image goes in as an `image_url` data URI, followed by the question.
- **sidecar:** a second sidecar is started from the production script on port 8182. Its `/describe` text replaces the
  image as the gateway does: a text part `[Attached image — described by the vision model (the main model cannot see
  pixels directly): <description>]`, followed by the question. Ornith answers from that text.

## Scoring, frozen

- A fact counts if any of its `|` alternatives occurs in the final answer. Matching is on lower-case text, with every
  dash folded to `-`.
- Score = facts found.
- **PASS iff native total ≥ sidecar total.**
- Reported per image; the total decides.
- Latency is reported, not gated.

```json
{
  "fixtures/newspaper.jpg": ["new york times", "1969", "men walk on moon", "flag", "eagle"],
  "fixtures/code.png": ["parse_invoice_total", "inv-2041", "startswith", "split", "return total"],
  "../../tests/vision-fixtures/glm53/screenshot.png": ["release-workspace", "build 42", "failed", "cuda_home", "deploy.sh"],
  "../../tests/vision-fixtures/glm53/diagram.png": ["document pipeline", "input", "encoder", "summary", "blue"],
  "../../tests/vision-fixtures/glm53/earth.jpg": ["earth", "africa", "cloud", "madagascar", "ocean"],
  "fixtures/vietnamese.png": ["2041", "hóa đơn|hoa don|invoice", "10 tháng 10|10/10|october 10|10 october", "2026", "1.250.000|1,250,000|1250000"]
}
```

Paths are relative to `speed-bench/ornith-vision/`.

## What this cannot say

- Six images, a single model, and one run per arm at temperature 0.
- A fact found is not a correct answer. A long description can name a fact and still get its context wrong.
