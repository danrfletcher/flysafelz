"""Fixtures for the online judge's tests (7 PR-3.F1). No model, no socket, no key: the judge is a fake callable object or 5a's own `Judge` on a fake chat model.

    judge_tree(key, path=, source=, lz_name=, run_id=, feedback=, question=)   a run-tree factory: a seed-style trace (guard record, `assess` and `search_guidance` runs) with the pilot's question in its inputs
    generate_trees(n, prefix=, **kw)   `n` such trees with keys `<prefix>-0` ... `<prefix>-<n-1>`: the run ids are uuid5 of the key on a fixed namespace, so every hash bucket is the same on every run
    judge_client(trees, **kw)          the online recording client over the trees, with each tree's `feedback` already stored on its run
    FakeJudge(scores=, raises=)        what the job calls as `judge.judge(example, answer)`: fixed per-criterion scores for the criteria of the example's path, None for the rest (5a's shape), and a record of its calls
    by_run(client)                     {run id: [feedback in the order stored]}

`feedback` is a list of (key, evaluator version) or (key, version, comment): what the run already has, as an earlier pass left it.
"""

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from online_helpers import FakeTree, OnlineRecordingClient, scored_tree

from evals.agent.evaluators.answer import ScoredAnswer
from evals.agent.judge import CRITERIA, criteria_for
from evals.agent.judge import feedback_key as judge_feedback_key
from evals.online import job

VERSION, OTHER_VERSION = "0123456789ab", "ba9876543210"
QUESTION = "Can I fly at the field tomorrow morning?"
BRIDGE_ERROR = "bridge returned 503"


def judge_tree(
    key: str,
    *,
    path: str = "assess",
    source: str = "pilot",
    lz_name: str | None = None,
    run_id: uuid.UUID | None = None,
    feedback: Sequence[tuple[Any, ...]] = (),
    question: str = QUESTION,
    **root_overrides: Any,
) -> FakeTree:
    """A seed-style trace with the pilot's question, for the judge to be asked about. `run_id` replaces the id made from `key`."""
    fields: dict[str, Any] = {"path": path, "source": source, **root_overrides}
    if lz_name is not None:
        fields["metadata"] = {"lz_name": lz_name, **fields.get("metadata", {})}
    if run_id is not None:
        fields.update(id=run_id, trace_id=run_id)
    tree = scored_tree(key, **fields)
    tree.root.inputs["messages"] = [{"type": "human", "content": question}]
    tree.feedback = list(feedback)
    return tree


def generate_trees(n: int, *, prefix: str = "gen", **kw: Any) -> list[FakeTree]:
    return [judge_tree(f"{prefix}-{index}", **kw) for index in range(n)]


def judge_client(trees: Iterable[FakeTree], **kwargs: Any) -> OnlineRecordingClient:
    trees = list(trees)
    client = OnlineRecordingClient([tree.root for tree in trees], children=[child for tree in trees for child in tree.children], **kwargs)
    for tree in trees:
        for key, version, *comment in tree.feedback:
            client.seed_feedback(tree.root, key, version=version, score=None if key == job.SAMPLED_KEY else 1, comment=comment[0] if comment else "earlier")
    return client


def by_run(client: OnlineRecordingClient) -> dict[Any, list[Any]]:
    found: dict[Any, list[Any]] = {}
    for posted in client.feedback:
        found.setdefault(posted.run_id, []).append(posted)
    return found


def posted_after(client: OnlineRecordingClient, before: int) -> list[Any]:
    """The feedback stored since `before` items were there."""
    return client.feedback[before:]


class FakeJudge:
    """5a's `Judge.judge(example, answer) -> {judge.<criterion>: {key, score, comment} | None}` with fixed scores. `scores` is {criterion: 1, 0 or None}; the default is 1 for every criterion.

    `raises` holds example ids (the run id as text) the judge raises ConnectionError on, as a bridge outage would; `drop` holds criteria it gives no result for (None, as a partial reply would).
    `calls` is every (example, answer) it was given, a raise included.
    """

    def __init__(self, scores: Mapping[str, int | None] | None = None, *, raises: Iterable[str] = (), drop: Iterable[str] = ()) -> None:
        self.scores = {name: 1 for name in CRITERIA} | dict(scores or {})
        self.raises, self.drop = set(raises), set(drop)
        self.calls: list[tuple[Mapping[str, Any], ScoredAnswer]] = []

    def judge(self, example: Mapping[str, Any], answer: ScoredAnswer) -> dict[str, Any]:
        self.calls.append((example, answer))
        if example["id"] in self.raises:
            raise ConnectionError(BRIDGE_ERROR)
        applicable = criteria_for(example["reference"]["path"])
        grades: dict[str, Any] = {judge_feedback_key(name): None for name in CRITERIA}
        for name in applicable:
            if name not in self.drop:
                grades[judge_feedback_key(name)] = {"key": judge_feedback_key(name), "score": self.scores[name], "comment": f"fake reason for {name}"}
        return grades

    def judged_ids(self) -> list[str]:
        return [example["id"] for example, _answer in self.calls]
