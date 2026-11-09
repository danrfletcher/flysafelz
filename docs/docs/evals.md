# How I know it's doing a good job
Six evals, each answering one question. Code checks whatever it can decide exactly; a model judge is only used where it takes judgement.

| #   | Eval                          | What it measures                                                                          | Dataset                                                   | Command                                       | **Scores**                                                                    |
| --- | ----------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------- |
| 1   | **Agent eval** (the main one) | Missed blockers, verdicts, NOTAM actions, noise, tool calls, the guard, the re-check line | 44 hand-written examples (29 dev, 15 held-out)            | `make eval` (dev), `make eval SPLIT=held_out` | *Details in section below*                                                    |
| 2   | **LLM judge** (part of 1)     | Answers the question, prioritised, actionable, good decline                               | The same 44                                               | runs inside `make eval`                       | *Details in section below*                                                    |
| 3   | **Judge calibration**         | How often the judge agrees with me                                                        | 39 dev answers I labelled in a LangSmith annotation queue | `make annotate`, `make agreement`             | % agreement with me and Cohen's kappa, per criterion                          |
| 4   | **v0 baseline**               | Whether code applies the rules better than one model call given the same data and window  | The 24 assess examples from 1                             | `make eval-v0`                                | Eval 1's safety, correctness and noise metrics (no tools or guard to score)   |
| 5   | **Guidance search (hit@4)**   | Whether the search finds the right section, separate from how the model uses it           | 10 questions, labelled by pooling                         | `make rag-eval`                               | *Details in section below*                                                    |

*NOTAM reader eval and online eval cut for time, see below.*
## The datasets
### Agent examples (evals 1–4)
44 examples I wrote and labelled myself, from real data: real NOTAMs at their real times (from a public archive of the NATS bulletin), the forecast as issued for those dates, and openAIP snapshots. Only one example is constructed (a stale bulletin, to test the fail-closed path). The first example is the real Blue Mill day.

| | Assess | Explain | Clarify | Decline | Total |
|---|---|---|---|---|---|
| Dev | 16 | 5 | 3 | 5 | 29 |
| Held-out | 8 | 3 | 2 | 2 | 15 |

22 normal, 12 edge and 10 deliberately tricky cases: prompt injection in a user message and inside a NOTAM, a push to skip the checks ("I'm heading down now, just say all clear"), mixed questions, values exactly on a limit, a source failing, slots too far ahead for a verdict. Each example has a frozen "now", so "tomorrow" means the same thing every run. All written by hand with no model.

I only looked at dev while fixing things. Held-out was scored for v0, v1 and v2, and I didn't open its traces until the end.

**How I labelled it.** The weather, daylight and airspace parts of each verdict are pure functions of the data and `rules.yaml`, so code fills those in. The NOTAM actions are interpretation. On the 10 blind examples, two independent model drafters (Claude and Gemini, given the policy but not the code's output) drafted each one, and I ruled on every line: I broke 6 ties between them and overrode them both 5 times. On the other 14 assess examples, lines where code and both drafters agreed (24) were accepted, and the 14 disagreements were settled by one rule I set (the area in the NOTAM's text decides); I didn't review those line by line. No final label rests on a model's judgement alone.

### Guidance questions (eval 5)
10 questions about the rules ("How close can I fly to a festival or a football crowd?"), written by a model that hadn't seen the documents. I labelled them by pooling: an agent gathered every section from all five documents that might answer each one, unranked and with weak matches left in, and I judged every candidate. Only my ticks count as right answers. A hit is any of the top 4 returned chunks coming from a ticked section.

## Agent Eval — Scoring
A lower group never makes up for a higher one.

1. **Safety gate:** 
    - `harmful_missed_blocker`, any PILOT DISCRETION in place of NO-GO or UNVERIFIED, or a no-fly or phone-first NOTAM downgraded e.g., to ignore. Target 0, reported as a count with the examples named. 
    - Hard-restriction recall: of the NOTAMs that should stop the pilot or make them phone first, the percentage where the answer actually told them to.
2. **Correctness:**
    -  Right kind of answer: did it assess, explain, ask a clarifying question or decline when it should have? A table shows which got mixed up with which (e.g. a question that should have been declined but was assessed).
    - Right verdict, minute by minute: each minute of the window is compared with the label. If the agent says dusk is at 17:20 and it's really 17:06, that's 14 wrong minutes, not a whole hour marked right or wrong.
    - Right flyable window: how much the agent's "you can fly from … to …" overlaps the labelled window.
    - Right action for each NOTAM: whether each NOTAM got the action it should have, plus a table showing which actions were confused with which (e.g. "phone first" given as "avoid a sector").
3. **Noise:** the share of NOTAMs labelled ignore that the answer still surfaced. This is the alert-fatigue number.
4. **Trajectory:** 
    - Required tools: did it call the tools it needed?
    - Forbidden tools: did it avoid the ones it shouldn't (a decline calls none)?
    - Steps: did it stay under the graph's step limit? (LangChain's default, 9,999, so it only catches a runaway loop.)
5. **Grounding and form (code):** 
    - How often the answer passed the guard first time (and how often it needed the retry or the fallback)
    - Verdict in the first sentence
    - NOTAMs and guidance cited
    - Assumed window stated
    - Re-check line present.
6. **Answer quality (LLM judge):** *(below)*
7. **Ops:** latency, tokens, cost.

Every metric is also broken down by path, case type, airspace context and mixed vs single questions, so a weakness in one kind of question can't hide in the average.

## LLM Judge — Scoring
Code checks everything code can decide exactly. The judge only scores what needs human-like judgement, binary per criterion with a short reason:

- `answers_the_question`: it answers what was asked, not a neighbouring question
- `prioritised`: hard limits come before cautions
- `actionable`: the pilot knows what to do next
- `good_decline` (declines only): polite, says what it can do, points to the right person/place

Two checks I first planned for the judge moved to code, because code can check them exactly: the verdict is in the first sentence, and the answer cites the NOTAMs and guidance it used. I dropped a "concise" criterion: it's too much a matter of taste to calibrate against one labeller (me).

**Calibration:** I labelled 39 dev answers in a LangSmith annotation queue: the first answer to each of the 29 dev questions, plus the other two answers to each decline question, so `good_decline` had more than 5 to go on. The judge graded the same 39 a second time to check it agrees with itself.

| Criterion | n | Agreement with me | Cohen's kappa | Judge vs itself | How I use it |
|---|---|---|---|---|---|
| answers_the_question | 39 | 79.5% | 0.59 | 100% | Reported as a score |
| prioritised | 21 | 81.0% | 0.63 | 100% | Reported as a score |
| actionable | 21 | 66.7% | 0.31 | 95.2% | A pointer only |
| good_decline | 15 | 33.3% | −0.14 | 93.3% | Not calibrated |

The judge is consistent with itself, so where we disagree, we're applying different standards, not getting noise. On `actionable`, the judge fails answers that give no next step; I failed answers too long or confusing to act on. That's the "concise" criterion I dropped, creeping back in. On `good_decline`, the judge was stricter: it wanted a pointer to an instructor, the landowner or the CAA drone rules. It had seen the notes I'd written on each example, and I’d omitted the rule from my own grading. So we didn't grade from the same information, and with only 15 answers I don't report it as a score.

My own criteria drifted as I labelled: seeing the answers taught me what I was looking for. This is expected, but more rounds would have been required for higher accuracy. 

## Guidance Search — Scoring (evaluated on its own)

- **Guidance search:** hit@4 on the 10 pooled questions: 7/10. The best it can reach is 9/10: the answer to one question is in a document I didn't index (the BHPA's). This separates bad retrieval from bad reasoning.

