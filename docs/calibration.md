# Calibrating the judge

The judge (`src/comfy_controller/adapters/claude_judge.py`) has never seen a
real image in this codebase -- every existing test drives it through a
scripted fake client (`tests/test_judge.py`). That proves the plumbing is
correct. It proves nothing about whether the judge's opinion tracks a
human's opinion on real, imperfect, AI-generated marketing images.

`src/comfy_controller/calibration.py` is the harness that measures that.
This document covers: how to build a labelled set worth measuring against,
how to run the harness, how to read what it prints, and the one rule for
whether the result clears you to let the judge auto-skip assets in
production.

If you take away one sentence from this whole document, take this one:
**a random sample of your usual output measures nothing.** Read "Building a
labelled set" before you start clicking through a folder of images.

## Building a labelled set

A labelled set is a directory of images plus a labels file that says, for
each image, whether a human considers it a QC pass or fail -- and, ideally,
*which* rubric check it violates when it fails.

### Deliberately oversample edge cases

Most of the images a real batch produces are obvious passes. If you label 40
images by grabbing whatever the last run produced, you'll get something like
35 obvious passes and 5 obvious failures -- and the judge will get all 40
right, because obvious cases are, definitionally, easy. You'll walk away
with a beautiful precision/recall table that tells you nothing about the
one thing you actually need to know: what does the judge do on the images
that are actually hard to call?

Instead, go looking for:

- **Borderline composition/lighting** -- not blown out, not pitch black,
  just "a person would hesitate here."
- **Subtle anatomy errors** -- a slightly-wrong hand, not a six-fingered
  one; these are the ones a fast human reviewer also misses on a first
  pass, which is exactly why they matter.
- **Near-brand palette** -- close to on-brand but with a cast that a brand
  reviewer would flag and a casual viewer wouldn't.
- **Partial artifacts** -- a small inpainting seam, one garbled word in
  background signage, not a screamingly obvious warped face.
- **Each required rubric check, individually** -- deliberately find (or
  stage) a handful of images that fail *only* `anatomy_correct`, only
  `matches_brief`, and so on, one check at a time. This is what makes
  per-check agreement (see below) meaningful instead of empty.
- **Both directions of every check** -- clear passes AND clear fails for
  each one, not just fails.

A good rule of thumb: if you can guess an image's label from across the
room without looking closely, it doesn't belong in the labelled set. It's
already telling you nothing.

### Labelling

Two label file formats are supported, either one paired with a directory of
images (`--images-dir`, or images resolved relative to the labels file's own
directory if you skip that flag):

CSV:

```csv
image,label,violated_checks
img001.png,pass,
img002.png,fail,anatomy_correct
img003.png,fail,anatomy_correct;no_artifacts
```

YAML:

```yaml
- image: img001.png
  label: pass
- image: img002.png
  label: fail
  violated_checks: [anatomy_correct]
- image: img003.png
  label: fail
  violated_checks: [anatomy_correct, no_artifacts]
```

`label` is `pass` or `fail` (also accepts `accept`/`reject`). `violated_checks`
is optional but valuable: without it, a "fail" image only contributes to the
overall accept/reject metrics, never to per-check agreement, because there's
no way to know *which* check should have caught it. Check names must match
`config/rubric.yaml`'s `name:` fields -- the harness warns (not errors) on a
name it doesn't recognise, which is usually a typo worth fixing before you
spend money on the run.

Thirty images is a bare minimum to get numbers at all. Fifty, oversampled
for edge cases as above, is a more honest starting point. Either way, expect
wide confidence intervals -- see below.

## Running it

Harness self-test, no API key, no network, no spend -- confirms the plumbing
(parsing, metrics, bootstrap, report writing) works before you trust it with
real money:

```bash
python -m comfy_controller.calibration \
  --labels labels.csv --images-dir images/ --fake
```

A real run, against the actual Claude judge:

```bash
python -m comfy_controller.calibration \
  --labels labels.csv --images-dir images/ \
  --repeats 3 --output-dir calibration_results/
```

This prints a cost estimate and asks for confirmation before it spends
anything:

```
Calibration cost estimate (upper bound -- the deterministic prefilter may reject
some images before the paid API is ever called, which this estimate assumes does
not happen):
  model: claude-opus-5
  42 images x 3 repeats x 3 internal self-consistency samples = 378 judge API calls
  ~1400 input tokens/image, ~700 output tokens/call, one prompt-cache warm-up assumed to cover the whole run
  ESTIMATED SPEND: $12.34
Proceed and spend an estimated $12.34? [y/N]
```

Pass `--yes` to skip the interactive prompt in a non-interactive context
(e.g. CI) -- only do this once you already trust the estimate, since it's
the last checkpoint before real spend.

`--repeats` (default 3, minimum 2) controls how many times the judge is
re-run on *each* image. This is separate from, and on top of, the rubric's
own `self_consistency.n` internal vote (`config/rubric.yaml`) -- `--repeats`
measures whether a full `judge.judge()` call, self-consistency vote and all,
gives the same final answer twice in a row on the same image. Fewer than 2
repeats makes that unmeasurable, which is why the harness refuses to run
with `--repeats 1`.

Output: `calibration_results.json` (everything, machine-readable) and
`calibration_summary.md` (the same content, formatted to read).

## Reading the numbers

### The headline is the interval width, not the point estimate

With 30-50 images, a reported precision of "0.80" is close to meaningless on
its own -- the honest version of that number is "0.80, 95% CI [0.45, 0.98]."
The summary prints the width right next to the point estimate and flags
anything wider than 0.25 as WIDE. Do not act on a point estimate whose
interval is wide; it will move the next time you add ten more labels.

### Self-consistency vs. human-disagreement

The summary reports two different disagreement rates side by side:

- **Self-inconsistency**: across the `--repeats` re-runs of the *same*
  image, how often did the judge change its mind?
- **Human-disagreement**: across images, how often did the judge's
  (majority-of-repeats) decision differ from the human label?

If the judge disagrees with itself almost as often as it disagrees with the
human, that's the important finding, not a footnote: it means a meaningful
share of the "wrong" verdicts were never resolvable by better labelling in
the first place, because the judge wasn't even consistent with itself on
the same pixels. The summary computes the maximum share of the
human-disagreement rate that self-inconsistency alone could explain, and
says so explicitly. When this share is large, the fix is prompt/rubric
work (tighter check wording, more grounding exemplars, possibly a different
`effort`/model), not more labels.

### Per-check agreement

Printed worst-first. A single badly-worded check dragging down the overall
number looks, from the aggregate metrics alone, exactly like "the model is
bad at QC." It usually isn't -- it's one ambiguous question. If one check
sits well below the others, reread it against the "HOW TO WRITE A GOOD
CHECK" guidance at the top of `config/rubric.yaml` before touching anything
else.

`false_reject` / `false_pass` on each check tell you which *direction* it's
wrong in: a check with a lot of `false_reject` is too strict (rejecting
good images), a lot of `false_pass` is too lax (missing real defects).

### Threshold sweep

`QCVerdict.passed` is a hard AND over every required check -- one failed
check rejects the asset (k=1). The sweep table shows what precision/recall
would look like if the cutoff were instead "k or more required checks must
fail" for every k up to the number of required checks. This is descriptive,
not prescriptive: it's the evidence you'd cite if you ever wanted to argue
for a different cutoff, not an automatic reconfiguration.

## What blocks a production run

The report ends with a PRODUCTION GATE section computed from a fixed,
written-down rule (`evaluate_production_gate` in `calibration.py`) -- not a
judgment call left to whoever's reading it that day. It is **BLOCKED**
unless all of the following hold:

1. **At least 30 labelled images** were usable in the decision metrics.
   Below that, no interval is worth reading.
2. **Precision and recall 95% CI width <= 0.25.** A wider interval means the
   sample hasn't resolved the question yet, whatever the point estimate
   says.
3. **Self-agreement across repeats >= 90%.** If the judge can't agree with
   itself at least this often, no amount of extra labelling will make its
   disagreement with humans fixable by labelling alone.
4. **No required check's agreement rate is below 70%.** A single bad check
   below this line means the aggregate numbers above are contaminated by
   one fixable rubric problem -- fix the check and re-run before trusting
   anything else.

If any of these fail, the summary prints exactly which one(s) and why,
under `PRODUCTION GATE: BLOCKED`. Do not override this by hand-waving "the
point estimate looked fine" -- the whole design of this harness is to stop
that exact move.
