# How it works
One LangChain `create_agent` call, which compiles to a LangGraph graph: a model and tools loop, plus one middleware. The below example follows one real question, my Blue Mill flight on 30th October:

1. A question comes in from the UI, LangGraph Studio or the eval harness. The coordinates come from the map click.
2. **Input check (code):** checks its a valid request, and a field in the UK.
3. **The model picks tools**, in any order: weather (Open-Meteo), airspace (openAIP), NOTAMs (via NATS pre-flight bulletin), and guidance search (RAG over the CAA and BHPA guidance).
4. **`assess` (code) sets the verdict** from the limits in [`rules.yaml`](https://github.com/danrfletcher/flysafelz/blob/main/src/flysafelz/rules.yaml) (e.g. mean wind: caution at 12 mph, no-go over 16 mph), each tagged with its source and how confident I am in it (sourced, best practice or educated guess), plus the airspace and each NOTAM's action. Those actions come from the NOTAM tool, which holds the only other model call: the **NOTAM reader** reads each NOTAM's free text into fixed fields, and code turns the fields into an action (see [what's code and what's the model](#whats-code-and-whats-the-model)).
5. **The model uses the data to answer the user’s question.**
6. **The guard (code)** checks every number, NOTAM ID, phone number and verdict word against the tool results. On a failure the model gets one retry with the finding. If that fails too, code shows the raw assessment instead. The pilot never gets an unchecked answer.
7. **Code adds** the limit lines ("Over limit: …") and the "re-check before you launch" line, and hydrates the redacted phone numbers back into place in the answer.

For Blue Mill the pilot gets PILOT DISCRETION – APPROACHING LIMITS · WITH CONDITIONS: the temporary danger area EGD496A active over the field with a number to phone first, the drone NOTAM H7171/26 0.65 NM south of the field up to 700 ft above ground (avoid that sector), and a wind value just over its caution limit.

## What's code and what's the model
I kept the model for the parts only a model can do. Everything that can be computed is code I can test.

| Job | Who |
|---|---|
| Choosing tools, writing the answer | Model |
| Reading what a NOTAM's free text means: heights in feet above ground, what kind of activity it is, exemptions, who to contact | Model (the **NOTAM reader**, one structured call per NOTAM, cached by ID) |
| Where a NOTAM's real area is (the coordinates in its text), distances, times, sunrise and sunset | Code |
| Each NOTAM's action (ignore, avoid a sector, phone first, check the supplement, no fly) | Code, from a fixed table over the reader's fields |
| The verdict | Code |
| Checking the answer | Code (the guard) |

The NOTAM tool's model call: fills fixed fields (heights, activity type, exemptions, which number is the contact, and whether the text gives a reason to stay out), and code turns those fields into the action with a fixed table. Its reading can change outcomes: a danger area over the field is "phone the operator first" unless the reader finds a stay-out reason in the text (live firing, entry prohibited, byelaws), which makes it "no fly". But it only ever acts through a field: an exemption counts only if its quote is really in the NOTAM, and there's no field for "this NOTAM doesn't matter". 

## StateGraph & Deep Agents
I used `create_agent` with one middleware rather than a hand-built `StateGraph`. The loop is the standard model and tools loop, and the safety lives in code around it (input check, guard, fallback, code-written lines). A hand-built graph would add nodes without adding behaviour. It would come in with the “watch” feature I cut (re-assess each hour, branch when the verdict changes, push a notification). That's a fixed, scheduled workflow rather than a conversation: code, not the model, decides each step, and the last verdict has to be kept between hourly runs to know whether anything changed. Explicit nodes and edges, saved state and scheduled runs are what a hand-built LangGraph graph is for.

LangGraph still does the work underneath: the checkpointer keeps each LZ's conversation (and the private phone-number map) between turns, `langgraph.json` serves the same graph to Studio, the UI and the eval harness, and the UI streams progress lines from it.

No Deep Agents: one field and one question is four or five tool calls in a short context, so there's nothing to plan, store in files or hand to a sub-agent, and each of those would make the trajectory harder to evaluate. If this grew into something like a week's paramotor adventure trip planner ("which of these 10 fields, which days can I fly and in which windows?"), a deep agent would plan the job and use today's agent as its sub-agent, so these evals would still hold.

