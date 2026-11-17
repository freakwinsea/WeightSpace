# Protocol and report templates

## protocol.md (write and push before the first forward call)
```markdown
# Exp NNN: <one-line question>

Written <date>, **before any <model> call**. Changes go in the change log at the bottom, with a reason. Harness: <files>, with WS_HERE set to this folder. Budget: <N> calls.

## Question
<What would a yes or a no tell us? Why this model?>

## What I already know (contamination)
<Published edits on this model, prior experiments, your priors. If you know the answer, the run is not blind: say so.>

## Procedure (fixed now)
1. Call 1 (generate): <calibration slice: problems (dataset, seed, n), cap, slots>.
2. Call 2 (forced): <direction measurement>.
3. The edit: <rule for layers, columns (K), strength (alpha), fixed now, nothing tuned later>.
4. Eval: <arms> on <problems (dataset, seed, n, disjoint from calibration)> × <seeds>, <cap>, <slots>.

## Predictions (written before any call)
- <layers, failure mode, effect size; each one checkable>

## Primary endpoint
For each problem, net = (edit correct over S draws) − (base correct over S draws).
Two-sided sign test over problems with net ≠ 0. **p < 0.05 = confirmed.** Scored once; n is fixed.
Secondary: accuracy, per-problem finish rate, looping, tokens on draws both arms solved, draw-level sign test (anti-conservative).

## Not done
<controls skipped and why>

## Change log
- <date>, **<phase, e.g. "during the eval, before any scoring">**: <what changed, why, what was already seen (nothing / summary lines only)>.
```

## Sign test and power sizing
```python
from math import comb, ceil

def sign_p(w, l):                       # two-sided exact binomial on discordant pairs
    n = w + l
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n) if n else 1.0

# Sizing from a calibration of base vs edit on one seed:
# d1 = share of problems where exactly one arm is correct.
d2 = 1 - (1 - d1) ** 2                  # per-problem discordance with 2 seeds (approximation)
n = ceil(40 / d2)                       # ~40 discordant problems gives ~80% power at a 7:2 split
```
- Refuse to size the test when base finishes fewer than 25% or more than 75% of the calibration problems; pick a different difficulty band instead. Near the floor or the ceiling, almost no pairs can change outcome.
- Calibration discordance can come in lower than planned. In Exp 003 the plan aimed for 40 discordant problems and got 28. If the calibration split looks weaker than the effect the test is sized for, write that down as a warning before the run.
- Cap n at a GPU-hours figure the user agrees to, and report the power you actually achieved.

## REPORT.md
```markdown
# Exp NNN report: <question>

<date>. Editor: <model>. Pre-registration and change log: `protocol.md`. Raw rows: `calls/...`. Budget: **X of N calls**.

## Result
**<One sentence: confirmed / not confirmed / mixed, and the headline number.>**
<Paired table: arms × correct, finished (correct), looping, tokens on both-solved.>
- Primary: <w / l, p>. <Confirmed or not, under the pre-registered rule.>
- Secondary: <...>

## Predictions vs outcome
| Prediction | Outcome |

## What it tells us about the mechanism
## Caveats
## For the user's hypothesis
## What would decide it (if not confirmed)
```
- Put the result first and say it plainly. Label every post-hoc number "post-hoc, descriptive only".
- Describe each dead end with the lesson it taught.
