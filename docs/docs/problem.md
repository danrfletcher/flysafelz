# The problem

## The problem in more detail
A paramotor pilot must consider a multitude of factors before deciding to fly:

- **Wind at the ground.** You launch and land on your feet, so mean wind and gusts set hard limits. The gap between gust and mean wind (gust spread) matters as much as the speed.
- **Wind higher up.** Wind that strengthens with height (the gradient) can make the climb-out rough, and strong wind aloft can stop you getting back to the field.
- **Rain, cloud and visibility.** A wet wing can stall, rain erodes the propeller, and low cloud or poor visibility take away the horizon.
- **Thermals and convection.** Strong sun and unstable air make turbulent midday air; very unstable air means showers and thunderstorms.
- **Daylight.** Flying stops at dusk.
- **Permanent airspace.** Controlled airspace around airports, aerodrome traffic zones and danger areas: fixed shapes on a chart, each with a floor and a ceiling.
- **NOTAMs.** Temporary changes to all of the above: a danger area switched on for the day, a drone survey, a display, a crane.
- (Out of scope here: the field itself, the landowner, the kit and the pilot.)

Everything above besides NOTAMs is a number against a limit or a shape on a map, so code can check it exactly. NOTAMs are the messy part: free text in an old telegraph style, where the important detail (the real area, the height, who to phone) is buried in the wording. There’s two places the model earns it’s place in this project and NOTAMs are one of them, and where most of the work went. More on that below.

### The anatomy of a NOTAM
*Firstly, 2 quotes from **Robert Sumwalt**, the former Chairman of the US National Transportation Safety Board (NTSB)*

> "That's what NOTAMs are: they're just a bunch of garbage that nobody pays any attention to." 
> “They're written in some kind of language that only a computer programmer would really understand. Just pages and pages and pages of irrelevant material, the important stuff gets buried.”

In light of the above, it goes without saying that NOTAMs are not in fact useless. They are one of the primary means of communicating safety information, and highly relevant to any pilot, not limited to paramotoring. Pilots don't skip NOTAMs because the information is missing. They skip them because of the noise. For instance, a NOTAM's header circle usually covers far more than the area it's actually about, so most of what a pilot scrolls past doesn't apply to them. The job isn't finding NOTAMs, it's deciphering which to discard, which to apply, and how to act upon them.

So alert fatigue is the real failure, and **precision is a safety metric**. A harmful error goes both ways: missing a restricted area, and crying wolf to the pilots.

## What the agent does and doesn't do
In scope: **whether it's safe and legal to fly from a given LZ, when, and how to interpret the data**. See [the dataset](evals.md#agent-examples-evals-14) for example questions.
Out of scope: take-off rules and guidance (i.e. landowner/land-law, surface, obstacles), ratings, drones, kit, flight paths, “watching” an LZ with notifications. The agent will **decline** anything in this category.

Why this shape?: **The brief asks for an agent a human would trust to take actions. Here the actions are things like:** phone the NOTAM operator first, stay below a given height, avoid a sector, or stay home. The agent only ever decides when *not* to fly, or which conditions should be attached to a flight. Flying is always the pilot's call once they're in the field, and hence the best case scenario is a pilot discretion verdict. For foot-launched aviation, deciding when *not* to fly is the decision that matters, and it's also the most difficult to get right.

Every answer carries one verdict per time segment, set by code:

| Verdict                                   | Means                                                                                                                                                 |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NO-GO**                                 | Something in the data categorically rules out the flight (wind over the limit, a restricted area active over the field)                               |
| **UNVERIFIED**                            | A source failed or a NOTAM over the field couldn't be read. Defers a manual check to the pilot                                                        |
| **PILOT DISCRETION – APPROACHING LIMITS** | No data categorically rules out the flight, but something's close: a value near its limit e.g., dusk, thermic air                                     |
| **PILOT DISCRETION – WITH CONDITIONS**    | No data categorically rules out the flight, but only if the pilot does something first: phone’s the NOTAM operator, stays below X ft, avoids a sector |
| **PILOT DISCRETION**                      | Nothing in the data rules it out, the pilot makes the call on the field                                                                               |

If a source fails, the answer is UNVERIFIED, never PILOT DISCRETION, and it says how to check by hand. There's no "GO". Decision support only, not a primary means of navigation.

