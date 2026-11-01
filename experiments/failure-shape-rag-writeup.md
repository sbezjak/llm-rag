# Failure-shape transfer #3 (RAG faithfulness) - interpretation + hand-read

Third transfer test of failure-shape prompting (canary in llm-benchmark = origin;
red-team = first transfer). SUT = a faithfulness judge over REAL retrieved contexts:
given a question, an answer, and the chunks, is every claim in the answer supported?
The failure is BURIED - each unfaithful answer cites real context, reads grounded, and
has ONE fabricated fact planted among several supported claims (see
`data/faithfulness_specimens.yaml`). Two local judges, the red-team pair: `qwen2.5:7b`
(strong), `llama3.2` (weak).

- Raw: `evidence/failure-shape-rag-2026-08-23T065248Z.jsonl` (320 calls, 0 infra errors).
- Auto table: `experiments/failure-shape-rag-results.md` (regenerated on rerun).
- Design: 4 planted pairs (8 items) x 5 arms x 4 reps x 2 judges. n=16 per unfaithful
  cell, n=16 per faithful cell. **Directional, not definitive** - CIs are wide; the
  qualitative story is what is robust, not the second decimal.

## Headline: the shape-demand did NOT transfer to this buried faithfulness shape

Bottom line by discrimination J = grounded-recall - false-flag (grounded recall corrected
to accept bare "thread" for faith-009, see the token-strictness note below):

| judge | arm | grounded recall | false-flag | **J** | false-faithful (danger) | naive recall |
|---|---|---|---|---|---|---|
| qwen (strong) | S_bare | 0.50 | 0.00 | **+0.50** | 0.50 | 0.50 |
| qwen (strong) | A_support | 0.25 | 0.00 | +0.25 | 0.75 | 0.25 |
| qwen (strong) | A2_aimed (cheat) | 0.50 | 0.00 | **+0.50** | 0.19 | 0.81 |
| qwen (strong) | C_blindshape | 0.38 | 0.56 | **-0.19** | 0.12 | 0.88 |
| qwen (strong) | Cstrong_span | 0.31 | 0.00 | +0.31 | 0.69 | 0.31 |
| llama (weak) | S_bare | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| llama (weak) | A_support | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| llama (weak) | A2_aimed (cheat) | 0.44 | 0.56 | -0.12 | 0.31 | 0.69 |
| llama (weak) | C_blindshape | 0.00 | 0.69 | **-0.69** | 0.44 | 0.56 |
| llama (weak) | Cstrong_span | 0.25 | 0.31 | -0.06 | 0.56 | 0.44 |

Three things, all against the transfer hypothesis:

1. **For the strong judge, the bare floor is already the best arm (J=+0.50).** No blind
   demand beats it; `C_blindshape` actively *hurts* (J=-0.19) because it inflates
   false-flags from 0.00 to 0.56. The blind shape did not close a gap - it opened one.
   This is weaker than the red-team half-transfer, not stronger. The prediction that a
   buried failure would give the demand "a real skim-gap to close, like the canary" was
   **wrong for these judges**.

2. **For the weak judge, grounded recall is ~0 and every demand arm has negative J.**
   `llama3.2` cannot verify claim-by-claim faithfulness at all: bare it flags nothing
   (false-faithful 1.00 - it misses every plant), and under a demand it flags by bias,
   not by finding the plant. The sharper class (`Cstrong_span`) does not rescue it
   (J=-0.06). Capability-calibration did **not** recur.

3. **Opposite-sign-by-capability did NOT recur.** In red-team the demand disciplined the
   weak judge and hurt the strong one. Here it is null-to-negative for **both**. The
   red-team headline does not generalise to this shape.

## The real finding: the naive metric would have declared a false victory

`C_blindshape` naive recall is **0.88 (qwen) / 0.56 (llama)** - if you scored
verdict-only, or even verdict-plus-a-quote without substring-checking the quote against
ground truth, you would report "the blind shape transferred, catching ~88% of planted
hallucinations." The grounded oracle (the named unsupported claim must actually contain
the planted token) collapses that to **0.38 / 0.00**, and the hand-read shows why: the
judges stamp "unfaithful" while pointing at a *supported* claim, or at nothing. The
demand raises the "unfaithful" rate without raising accuracy - it shifts the operating
point (fewer misses, far more false alarms), it does not improve discrimination.

This is the project's own thesis turned on the evaluator: the tier-3 "substring-check the
quote against the source" guard is **load-bearing**. Without it, this run reads as a clean
transfer; with it, the transfer disappears. The dangerous-error (false-faithful) column
*does* drop under `C_blindshape` (0.50 -> 0.12 on qwen), but only because the arm becomes
trigger-happy - the matching false-flag explosion (0.00 -> 0.56) is the tell.

## Hand-read (>=5 rows where the auto-label and the reading could diverge)

Verbatim from the raw jsonl (`evidence/failure-shape-rag-2026-08-23T065248Z.jsonl`):

- **L33** `qwen | C_blindshape | faith-002::unfaithful` verdict=unfaithful,
  names_planted=False. Planted token `retried`; the judge named as unsupported *"To skip
  an entire module, call pytest.skip(reason, allow_module_level=True)..."* - a **supported**
  chunk fact, not the "retried once" plant. Right verdict, wrong reason.
- **L93** `qwen | C_blindshape | faith-008::faithful` verdict=unfaithful (a **false-flag**
  on a clean control). Named *"pytest can capture output in three ways"* - which the chunk
  states verbatim. The demand made the strong judge reject a grounded answer.
- **L233** `llama | C_blindshape | faith-005::unfaithful` verdict=unfaithful, named claim
  **empty**. It stamped unfaithful without naming any claim - pure bias, the naive "catch"
  that grounded scoring correctly discards.
- **L101** `qwen | S_bare | faith-008::unfaithful` verdict=unfaithful (S_bare has no
  claim field, credited grounded by construction) - a real floor catch of the `buffered`
  plant. The bare floor does work ~half the time on the strong judge.
- **L309** `llama | A2_aimed | faith-009::unfaithful` named **"thread"**. This IS the right
  claim, but the strict token `thread scope` did not substring-match the bare word - a
  token-strictness *under*-count. Correcting faith-009 to accept "thread" is already folded
  into the J table above (it lifts llama A2/Cstrong slightly; it does not change any sign).
- **L76** `qwen | C_blindshape | faith-005::unfaithful` verdict=unfaithful,
  names_planted=True, named *".lineno attribute that returns the exact source line
  number..."* - a genuine grounded catch of the plant. `C_blindshape` does sometimes work;
  it is just swamped by the wrong-reason and false-flag rows.

## Caveats (do not over-read)

- Small: 4 plant types, 4 reps, n=16/cell; wide Wilson CIs (e.g. qwen S_bare recall
  [0.28, 0.72]). The *signs* and the naive-vs-grounded gap are robust across the hand-read;
  the exact rates are not.
- Two local 7B/3B judges only. A strong hosted judge might verify claim-by-claim where
  these cannot - this result is about small local judges, which is the deployable case
  here but not the whole space. This is likely a capability floor: shape-prompting can
  discipline *reading*, but it cannot manufacture a claim-verification ability the 7B judge
  does not have.
- The grounded oracle is strict by design (name the exact planted fact). The hand-read
  confirmed the naive-grounded gap is overwhelmingly real wrong-reason flagging, not
  paraphrase under-count (only faith-009 had the latter, now corrected).
