# Ornith native vision vs the describe sidecar: result (2026-10-10)

The rule was frozen in `QUALITY-PRE-REGISTRATION.md` (commit `7ebba3b0`) before either arm ran. Raw numbers are in
`quality-2026-10-10.json`; the script is `quality.py`.

## Verdict

| | |
|---|---|
| native facts found | **29 / 30** |
| sidecar facts found | **29 / 30** |
| rule | native ≥ sidecar |
| outcome | **PASS** (a tie) |

## Per image

| image | native | sidecar | native answer | sidecar describe + answer |
|---|---|---|---|---|
| newspaper.jpg | 5/5 | 5/5 | 5.4 s | 18.9 s (includes loading the 4B model) + 4.7 s |
| code.png | 5/5 | 5/5 | 3.0 s | 4.1 + 2.6 s |
| screenshot.png | 5/5 | 5/5 | 8.2 s | 7.1 + 2.5 s |
| diagram.png | 5/5 | 5/5 | 6.1 s | 4.2 + 2.1 s |
| earth.jpg | 4/5 | 4/5 | 2.5 s | 2.9 + 2.9 s |
| vietnamese.png | 5/5 | 5/5 | 0.9 s | 3.3 + 2.7 s |

- **The fact both arms missed** is the same one: "madagascar" in the Earth photo.
- **The Vietnamese invoice.** Native answered in 67 characters and still contained all five facts. The sidecar route's
  answer is 749 characters, built from a 1,231-character description.

## What it says, and what it does not

- **Quality is a tie on this set, at the ceiling.** Both arms find almost every fact, so six images cannot
  separate them. The rule asked only that native not be worse, and it is not.
- **Native removes the describe step.** On the sidecar route the describe step alone takes 3-19 s per image before
  Ornith starts. Over the six images:
  - native answers took 0.9-8.2 s in total;
  - the sidecar route took 5.0-23.6 s.
- **Native also drops the 3.1 GB sidecar** and its 900 s idle unload/reload cycle, for 0.9 GB of mmproj inside the ds4
  process.
- **The middleman is gone.** The 35B model reads the pixels itself instead of a 4B paraphrase. This set does not
  show that as a quality gain; it shows no loss.
