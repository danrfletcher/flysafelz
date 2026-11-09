# Results

## Results
Models: agent Claude Sonnet 5.5, NOTAM reader Claude Haiku 5.5, judge Claude Opus 5.5. Mean ± spread over 3 runs. Full reports with per-slice tables and trace links: `evals/results/`.

### Was the design right? v0 vs v1

v0 is a rules-only baseline: one model call given the same data as v1 (the same prefilter, the limits, the NOTAM action table) and asked for the same outputs. In v0 the model does the arithmetic and applies the rules; in v1 code does. v0 is also handed the labelled window and never sees the pilot's question, so it skips the steps v1 has to get right first: deciding to assess, turning "later tonight" into hours, and getting an answer past the guard. It tests the rules step, not the whole agent.

| Metric (held-out)                          | v0            | v1 (before the v2 fix) |
| ------------------------------------------ | ------------- | ---------------------- |
| harmful_missed_blocker                     | 33.3% of runs | 62.5% of runs          |
| hard-restriction recall                    | 83.3%         | 10.0%                  |
| verdict match (minutes)                    | 99.9%         | 42.9%                  |
| verdict match, only runs where v1 assessed | 99.9%         | 75% (12 of 21 runs)    |
| noise rate                                 | 1.9%          | 3.9%                   |

v0 won, and on the rules step it matched or beat code. v1 lost its points before the rules ran. In 9 of its 21 held-out assess runs it never produced an assessment (the floor crash, a clarifying question instead of assuming a window, or a guard fallback), and each of those scores zero for every minute. So the architecture doesn't buy accuracy on the rules: a model compares a gust with 6 mph about as well as code. It buys guarantees: code gives the same answer every time, it's tested, and the guard can hold the model to it. A model at 99.9% is wrong now and then in ways nothing catches. The v2 fix closed much of the gap. On held-out, v1's verdict match rose from 42.9% to 71.4% (83% on the runs where it assessed) and hard-restriction recall from 10% to 30%. On dev, every v2 run that reached an assessment matched the labels on every minute (31 runs). v0 still leads on held-out. What's left is v1 not reaching an assessment (#3 below), a downgraded NOTAM (#2), and one example, ho_a3, where the NOTAM reader made a whole window NO-GO that the labels say is flyable with a sector to avoid (see Where it struggles).

### Did the evals improve it? v1 vs v2

| Metric                                           | v1 dev                    | v2 dev                                                                | v1 held-out                                                           | v2 held-out                                                           |
| ------------------------------------------------ | ------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| harmful_missed_blocker                           | 37.5% of runs             | 37.5%                     | 62.5%                     | 50.0%                     |
| missed blockers after I read every flagged trace | not checked               | 25% (10 of 40)            | not checked               | 25% (6 of 24)             |
| hard-restriction recall                          | 0%                        | 0%                        | 10.0%                     | 30.0%                     |
| path accuracy                                    | 87.4%                     | 87.4%                     | 82.2%                     | 82.2%                     |
| verdict match (minutes)                          | 51.5%                     | 89.0%                     | 42.9%                     | 71.4%                     |
| noise rate                                       | 9.2%                      | 14.1%                     | 3.9%                      | 3.9%                      |
| guard failures / fallbacks                       | passed first time 33.3%   | 33.3%                     | 37.8%                     | 35.6%                     |
| judge: answers / prioritised / actionable        | 54% / 64% / 37%           | 72% / 73% / 56%           | 58% / 64% / 48%           | 69% / 64% / 39%           |
| tokens and cost per answer                       | 17.5k in, 1.8k out, $0.05 | 16.3k in, 1.7k out, $0.04 | 15.7k in, 1.7k out, $0.04 | 13.4k in, 1.4k out, $0.04 |

Held-out proves the fix generalises. A fix that only moves dev was memorised, not learned.

It did generalise: verdict match rose on both splits, and held-out also gained on safety (recall 10% → 30%, one example no longer missing a blocker). Missed blockers didn't move on dev, because what's left there isn't the crash (see Where it struggles). The guard's first-time pass rate didn't move either, as expected: the guard wasn't changed. Actionable fell 9 points on held-out, about one answer in 11 per repeat, inside the spread between repeats.

By slice (dev, v1 → v2):

| Slice | n | verdict match | missed blockers |
|---|---|---|---|
| edge | 8 | 74% → 97% | 14% → 14% |
| normal | 15 | 60% → 79% | 40% → 40% |
| tricky | 6 | 7% → 89% | 75% → 75% |

The fix landed mostly on the tricky questions, where verdict match went from 7% to 89%. Missed blockers held in every slice, because the remaining misses are the vague-time and downgrade failures and the scoring gap, none of which the fix touched. The slices are small: one tricky example is 17 points.

## How the evals changed what I built
v1 was a good-faith first version: everything I'd designed, nothing held back to "discover" later. v2 contains only changes chosen after reading v1's results and traces. I ranked failures by severity (missed blockers first), then by how often they happened per slice, walked their traces to find which layer failed (prefilter, reader, rules, agent, guard), and fixed the most valuable one.

Every candidate fix from v1 dev (87 runs), ranked by severity, then by how many examples it hit:

| # | Failure | How it surfaced | Examples | Severity | Layer | Fix size |
|---|---|---|---|---|---|---|
| 1 | A NOTAM read with no floor crashes assess, so the answer falls back and blocking NOTAMs are lost | Agent eval: missed blockers and guard fallbacks clustered on the same 5 examples in every repeat; the trace showed the assess tool failing | 5 (3 with missed blockers) | Safety | Code | 1 line |
| 2 | A phone-first NOTAM read as awareness (dev_12 H2483 every repeat, dev_14 H1633 once) | Agent eval: the missed-blocker comment names the NOTAM and the action it was given | 2 | Safety | Reader / action table | Needs a trace read |
| 3 | A vague time ("later tonight", "tonight") gets a question back instead of an assumed window, so no checks run | Agent eval: missed blockers alongside wrong-path and required-tools failures on the same examples | 2 | Safety | Agent prompt | Small |
| 4 | The guard rejects true statements: a document number ("1535"), "a few minutes", the pilot's own times, a restated limit. The fallback it forces is a raw data dump | Guard pass at 33%; I grouped the guard's own findings, logged on each trace, by type | ~15 | Quality | Guard | Medium; it loosens a safety net |
| 5 | NO-GO headlines the whole window when a flyable slot sits inside it | Labelling the dev answers to calibrate the judge, then comparing with its grades: I failed 3 answers the judge passed | 3 seen | Over-caution | Agent prompt | Small, plus a new code check |
| 6 | A mixed question ("after shopping, and what's that airfield to the west?") gets the clarifying question but the airfield half is dropped | Agent eval: required tools at 0% in the clarify slice (dev_17, the only clarify example with a required tool) | 1 | Low | Agent prompt | Small |
| 7 | Declines don't point to a human (an instructor, the landowner, the CAA drone rules) | Judge: its `good_decline` reasons; my labels didn't confirm the standard | 5 | Low | Agent prompt | Small |

v2 fixes #1 only: one change per version, so a regression can be traced to its cause. #2 and #3 would be next.

The guard also did its job. It caught real model errors: "over the 16 mph limit" for a value of 5.8 mph, NO-GO on an outlook hour with no verdict, a verdict assess never returned, and the real date written into an example with a frozen date.

**The v2 Fix in More Detail: **A NOTAM read with no floor crashed the whole assessment.** In 5 of the 29 dev questions (Blue Mill and Druridge Bay), every repeat fell back to the "couldn't check" answer. The trace showed the assess tool failing. At Blue Mill the cause was Exercise Cobra Warrior, a NOTAM covering three danger areas from 250 ft up. openAIP had no shape for one of them, so code used the NOTAM's circle instead, and took its floor from the reader. The reader left the floor empty, and code had no fallback, so the whole assessment failed: weather and every other NOTAM were lost with it. My test fixtures always had a floor, so the tests missed it. The fix is one line: with no floor, the circle comes down to the surface, which is the cautious side. In v2 those 5 examples reach a real assessment: verdict match rose from 51.5% to 89.0% on dev and from 42.9% to 71.4% on held-out.

