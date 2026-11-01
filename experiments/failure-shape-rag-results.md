# Failure-shape transfer #3 (RAG faithfulness) - results

- judges: `qwen2.5:7b` (strong), `llama3.2` (weak)  temp: 0.7  reps/cell: 4
- raw: `evidence/failure-shape-rag-2026-08-23T065248Z.jsonl`  |  wall: 5646s
- items: 4 faithful controls, 4 unfaithful (one planted unsupported claim each), over real known-good contexts

> The failure is BURIED on purpose: each unfaithful answer cites real context and reads grounded, with ONE fabricated fact among several supported claims. A skim stamps it faithful. See `data/faithfulness_specimens.yaml` for how each plant was built and `planted_token` for the deterministic grounded check.

## Primary metrics (grounded scoring)

### Recall on unfaithful items (planted claim caught) - higher is better

grounded = verdict-right AND the arm named the ACTUAL planted claim (primary); naive = verdict-only (secondary). On S_bare / A_support the two are equal by construction (no claim-naming field to check).

| judge | arm | grounded k/n | grounded rate | grounded 95% CI | naive k/n | naive rate |
|---|---|---|---|---|---|---|
| qwen2.5:7b (strong) | S_bare | 8/16 | 0.50 | [0.28, 0.72] | 8/16 | 0.50 |
| qwen2.5:7b (strong) | A_support | 4/16 | 0.25 | [0.10, 0.49] | 4/16 | 0.25 |
| qwen2.5:7b (strong) | A2_aimed | 8/16 | 0.50 | [0.28, 0.72] | 13/16 | 0.81 |
| qwen2.5:7b (strong) | C_blindshape | 6/16 | 0.38 | [0.18, 0.61] | 14/16 | 0.88 |
| qwen2.5:7b (strong) | Cstrong_span | 5/16 | 0.31 | [0.14, 0.56] | 5/16 | 0.31 |
| llama3.2 (weak) | S_bare | 0/16 | 0.00 | [0.00, 0.19] | 0/16 | 0.00 |
| llama3.2 (weak) | A_support | 0/16 | 0.00 | [0.00, 0.19] | 0/16 | 0.00 |
| llama3.2 (weak) | A2_aimed | 4/16 | 0.25 | [0.10, 0.49] | 11/16 | 0.69 |
| llama3.2 (weak) | C_blindshape | 0/16 | 0.00 | [0.00, 0.19] | 9/16 | 0.56 |
| llama3.2 (weak) | Cstrong_span | 4/16 | 0.25 | [0.10, 0.49] | 7/16 | 0.44 |

### False-faithful on unfaithful items (planted claim MISSED, stamped faithful) - the DANGEROUS error, lower is better

| judge | arm | k/n | rate | 95% CI |
|---|---|---|---|---|
| qwen2.5:7b (strong) | S_bare | 8/16 | 0.50 | [0.28, 0.72] |
| qwen2.5:7b (strong) | A_support | 12/16 | 0.75 | [0.51, 0.90] |
| qwen2.5:7b (strong) | A2_aimed | 3/16 | 0.19 | [0.07, 0.43] |
| qwen2.5:7b (strong) | C_blindshape | 2/16 | 0.12 | [0.03, 0.36] |
| qwen2.5:7b (strong) | Cstrong_span | 11/16 | 0.69 | [0.44, 0.86] |
| llama3.2 (weak) | S_bare | 16/16 | 1.00 | [0.81, 1.00] |
| llama3.2 (weak) | A_support | 16/16 | 1.00 | [0.81, 1.00] |
| llama3.2 (weak) | A2_aimed | 5/16 | 0.31 | [0.14, 0.56] |
| llama3.2 (weak) | C_blindshape | 7/16 | 0.44 | [0.23, 0.67] |
| llama3.2 (weak) | Cstrong_span | 9/16 | 0.56 | [0.33, 0.77] |

### False-flag on faithful controls (grounded answer called unfaithful) - lower is better

| judge | arm | k/n | rate | 95% CI |
|---|---|---|---|---|
| qwen2.5:7b (strong) | S_bare | 0/16 | 0.00 | [0.00, 0.19] |
| qwen2.5:7b (strong) | A_support | 0/16 | 0.00 | [0.00, 0.19] |
| qwen2.5:7b (strong) | A2_aimed | 0/16 | 0.00 | [0.00, 0.19] |
| qwen2.5:7b (strong) | C_blindshape | 9/16 | 0.56 | [0.33, 0.77] |
| qwen2.5:7b (strong) | Cstrong_span | 0/16 | 0.00 | [0.00, 0.19] |
| llama3.2 (weak) | S_bare | 0/16 | 0.00 | [0.00, 0.19] |
| llama3.2 (weak) | A_support | 0/16 | 0.00 | [0.00, 0.19] |
| llama3.2 (weak) | A2_aimed | 9/16 | 0.56 | [0.33, 0.77] |
| llama3.2 (weak) | C_blindshape | 11/16 | 0.69 | [0.44, 0.86] |
| llama3.2 (weak) | Cstrong_span | 5/16 | 0.31 | [0.14, 0.56] |

## Quote validity (recursion guard - demanded quote actually in its source)

A confident but hallucinated quote is a compliance failure (the judge inventing evidence): a support span not in the context, or an 'unsupported claim' not in the answer.

### Valid quotes / demanded quotes

| judge | arm | k/n | rate | 95% CI |
|---|---|---|---|---|
| qwen2.5:7b (strong) | S_bare | 0/0 | 0.00 | [0.00, 0.00] |
| qwen2.5:7b (strong) | A_support | 22/32 | 0.69 | [0.51, 0.82] |
| qwen2.5:7b (strong) | A2_aimed | 13/32 | 0.41 | [0.26, 0.58] |
| qwen2.5:7b (strong) | C_blindshape | 17/32 | 0.53 | [0.36, 0.69] |
| qwen2.5:7b (strong) | Cstrong_span | 5/32 | 0.16 | [0.07, 0.32] |
| llama3.2 (weak) | S_bare | 0/0 | 0.00 | [0.00, 0.00] |
| llama3.2 (weak) | A_support | 18/32 | 0.56 | [0.39, 0.72] |
| llama3.2 (weak) | A2_aimed | 20/32 | 0.62 | [0.45, 0.77] |
| llama3.2 (weak) | C_blindshape | 5/32 | 0.16 | [0.07, 0.32] |
| llama3.2 (weak) | Cstrong_span | 4/32 | 0.12 | [0.05, 0.28] |

## How to read this

- **Grounded is the metric that counts.** A demand-arm 'catch' only counts if the judge named the ACTUAL planted claim (its named unsupported claim contains the item's `planted_token`). A large naive-vs-grounded gap means the judge stamped 'unfaithful' while pointing at the wrong claim (or none) - right verdict, wrong reason. Read the grounded column.
- **Transfer** = a blind shape-demand (C_blindshape) lifts GROUNDED recall toward the aimed cheat (A2_aimed) and above the bare floor (S_bare), WITHOUT blowing up false-flag, per judge. C ~ A2 = the shape transferred to a buried, non-adversarial failure. C ~ S = it did NOT (a real negative worth knowing).
- **Capability-calibration** = the weak judge needs the SHARPER shape (Cstrong_span) where the strong judge already wins on the loose one (C).
- **Backfire** = A_support under-flags (anchors on a supported span), recall at or below S_bare.
- **Opposite-sign-by-capability (the red-team headline - does it recur?)** = the demand helps the WEAK judge and is null/negative for the STRONG one. Report per judge whether that pattern holds on this third shape.
- **BOUNDARY NULL (report honestly)** = if S_bare false-faithful is already ~0, this plant is too salient to be a skim-trap for that judge; the demand has no gap to close there. That is a real negative, not a failure - do NOT force a gap.

## TODO before trusting this (CLAUDE.md recursion-read)

- Hand-read >=5 rows in the jsonl where an arm FLIPPED the label vs another arm (same judge+item), quote them verbatim file:line. Auto-coding `verdict==label` is fooled by right-verdict/wrong-reason (naive-caught but names_planted=false). If reading and the boolean diverge, the reading wins. Record the read sample in the writeup.
