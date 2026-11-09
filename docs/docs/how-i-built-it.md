# How I built it

## How I built it
I used coding assistants heavily, as the brief invites. I designed the agent, the verdict rules, the NOTAM action table, the eval design and the metrics, and I wrote and labelled the dataset by hand. For the guidance search eval I used pooling: a model wrote the questions without seeing the documents, an agent gathered every section from all five documents that might answer each one (unranked, weak matches included), and I judged every candidate. Only my ticks count as right answers. This stops me missing correct sections, which would unfairly lower the score, without anchoring me on a model's picks. 

The code was written by a build pipeline I had already wirtten previously to this project: a python state machine dispatches a coder, reviewer and tester for each ticket into an isolated git worktree. I scoped the project at a high level, split the build into 17 tickets then worked with an agent for each ticket to extract ambiguity and spec the technical implementation including the golden path, human tests for critical paths, required testing and handling of edge cases. 

What I checked myself: every decision in the spec, the labels (all handwritten), and the agent's graph and guard. I also ran the eval loop: ranked every failure v1 showed, fixed only the most severe (logged the rest), then checked that v2 moved the held-out dataset, not just dev. An assistant read most of the flagged traces and I questioned each finding until I could explain it. Before submitting, I ran an end-to-end smoke test as a pilot would: picked a field on the map, asked in the chat, and checked the answer against its trace.

As mentioned at the start, the agent itself is deliberately small. Most of the code is the deterministic core and the guard, and about 70% of the Python is tests. I wanted the safety critical behaviour somewhere I could test it rather than left to the model as I plan to deploy this in real life at my local paramotoring club.

## Beyond paramotors
The pattern is structured metadata wrapped around free text, where the cost of noise is that people stop reading. NOTAMs are one case; maintenance logs and compliance notices are others.

