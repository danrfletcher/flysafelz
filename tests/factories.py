"""Builders for the NOTAM action tests: LZs, segments, parsed NOTAMs, readings, shapes.

Parsed NOTAMs come from PR-4's `parsed()` (tests/unit/pib_helpers.py); readings are written by
hand. No fixture file, no network, no .env.

Guidance search tests (index, search_guidance, eval, gate) build their snapshots here too:
    FakeEmbedder                  bag-of-words hashing into a fixed-dimension vector (tests/guidance_helpers.py), so cosine ranking can be predicted; records its calls
    EmbedderSpy                   an embedder factory that counts how many times it was asked to build the embedder
    guidance_chunk(...)           one chunk of a document
    write_snapshot(folder, ...)   a data folder as `make guidance` leaves it (manifest.yaml + embeddings.sqlite, and optionally sources.yaml), built by PR-7's own writers

The judge calibration tests (5a PR-8.F1) build a fake v1 experiment here:
    fake_experiment(name, ...)    the LangSmith project and the root runs the v1 runner logs for the committed DEV examples (29 examples x 3 reps = 87 runs), as real
                                  langsmith schema objects; `errored` and `missing` name (example id, rep) slots; `splits` overrides a run's metadata split
    judge_feedback(run, criterion, passed)   the feedback the judge logs on a run (`judge.<criterion>`, source model)
    dan_feedback(run, key, value, ...)       a label Dan submits in the queue (the bare criterion name, source app)
"""

import hashlib
import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from guidance_helpers import FakeEmbedder
from langsmith import schemas as ls_schemas
from pib_helpers import parsed as pib_parsed
from weather_helpers import calm

from evals.agent.loader import example_uuid, load_dataset
from flysafelz.agent.settings import SETTINGS
from flysafelz.assess import assess, default_window
from flysafelz.clients.open_meteo import Forecast
from flysafelz.clients.openaip import Airspace, AirspaceFetch
from flysafelz.contracts import (
    AppliesTo,
    Assessment,
    Category,
    Circle,
    Contact,
    ContactKind,
    Datum,
    Exemption,
    ExemptionCovers,
    Limit,
    LimitUnit,
    NotamReading,
    ParsedNotam,
    Polygon,
    ProhibitiveReason,
    SourceName,
    SourceState,
    SourceStatus,
)
from flysafelz.geo import distance_nm
from flysafelz.guidance import corpus
from flysafelz.guidance.chunker import Chunk
from flysafelz.notam_action import ActionSegment, LandingZone, notam_action
from flysafelz.notam_area import NoArea, Shape, area_from_item_e
from flysafelz.pib import PibResult
from flysafelz.rules import RULES, Rules

BLUE_MILL = LandingZone(lat=53.802797, lon=-0.086644, elevation_ft=30.0)
LZ = BLUE_MILL
STANDARD_HPA = 1013.25
ARC_MINUTES = 60  # minutes of latitude in a degree: one per nautical mile


def utc(hour: int, minute: int = 0, day: int = 31, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def segment(start: datetime | None = None, end: datetime | None = None, pressure: float | None = STANDARD_HPA) -> ActionSegment:
    """A Saturday morning at Blue Mill by default: 09:00-10:00Z on 31 Oct 2026."""
    return ActionSegment(start=start or utc(9), end=end or utc(10), pressure_msl=pressure)


def with_rules(**values: object) -> Rules:
    """A copy of the shipped rules with the named rules' values replaced."""
    changed = {name: getattr(RULES, name).model_copy(update={"value": value}) for name, value in values.items()}
    return RULES.model_copy(update=changed)


# --- limits and places ------------------------------------------------------------


SFC = Limit(value=0, unit=LimitUnit.FT, datum=Datum.SFC)


def amsl(feet: float) -> Limit:
    return Limit(value=feet, unit=LimitUnit.FT, datum=Datum.AMSL)


def agl(feet: float) -> Limit:
    return Limit(value=feet, unit=LimitUnit.FT, datum=Datum.AGL)


def fl(level: float) -> Limit:
    return Limit(value=level, unit=LimitUnit.FL, datum=Datum.AMSL)


def offset(lat: float, lon: float, bearing: float, nm: float) -> tuple[float, float]:
    """The point nm nautical miles from (lat, lon) on a bearing, on the same flat map geo.py uses (checked to 1 %)."""
    theta = math.radians(bearing)
    point = (lat + nm * math.cos(theta) / ARC_MINUTES, lon + nm * math.sin(theta) / ARC_MINUTES / math.cos(math.radians(lat)))
    assert abs(distance_nm(lat, lon, *point) - nm) <= 0.01 * nm
    return point


def circle_at(edge_nm: float, bearing: float = 180.0, radius_nm: float = 0.5) -> Circle:
    """A circle whose edge is edge_nm from the LZ on the bearing. 0 puts the LZ at its centre (over it)."""
    if edge_nm == 0:
        return Circle(lat=LZ.lat, lon=LZ.lon, radius_nm=radius_nm)
    lat, lon = offset(LZ.lat, LZ.lon, bearing, edge_nm + radius_nm)
    return Circle(lat=lat, lon=lon, radius_nm=radius_nm)


def square_at(edge_nm: float, bearing: float = 180.0, half_nm: float = 0.5) -> Polygon:
    """A square whose nearest side is edge_nm from the LZ on the bearing (0: the LZ is in its middle)."""
    centre = (LZ.lat, LZ.lon) if edge_nm == 0 else offset(LZ.lat, LZ.lon, bearing, edge_nm + half_nm)
    dlat = half_nm / ARC_MINUTES
    dlon = dlat / math.cos(math.radians(centre[0]))
    ring = [(centre[0] - dlat, centre[1] - dlon), (centre[0] - dlat, centre[1] + dlon), (centre[0] + dlat, centre[1] + dlon), (centre[0] + dlat, centre[1] - dlon)]
    return Polygon(points=ring)


# --- NOTAMs -----------------------------------------------------------------------


def notam(notam_id: str = "H7171/26", *, circle: Circle | None = None, **overrides: object) -> ParsedNotam:
    """PR-4's parsed NOTAM, active all day 31 Oct. circle sets its Q-circle (centre and radius).

    Its E text draws no area, so where it is comes from the Q-circle until a test writes a shape into item E (see `drawn`).
    """
    if circle is not None:
        overrides.update(centre_lat=circle.lat, centre_lon=circle.lon, radius_nm=circle.radius_nm)
    return pib_parsed(notam_id, **overrides)


MICRO = 1_000_000  # seconds are written to six decimals, so a drawn shape comes back within a few millimetres


def dms(value: float, hemispheres: str, degree_digits: int) -> str:
    """One coordinate half as a NOTAM writes it: DDMMSS.ffffffN or DDDMMSS.ffffffW (hemispheres is \"NS\" or \"EW\")."""
    micro_seconds = round(abs(value) * 3600 * MICRO)
    degrees, rest = divmod(micro_seconds, 3600 * MICRO)
    minutes, seconds = divmod(rest, 60 * MICRO)
    return f"{degrees:0{degree_digits}d}{minutes:02d}{seconds / MICRO:09.6f}{hemispheres[0] if value >= 0 else hemispheres[1]}"


def coordinate(lat: float, lon: float) -> str:
    return f"{dms(lat, 'NS', 2)} {dms(lon, 'EW', 3)}"


def drawing(shape: Circle | Polygon) -> str:
    """The words an E text uses to draw this shape: a hyphen-joined list closed on its first corner, or WI <r>NM RADIUS OF <coordinate>."""
    if isinstance(shape, Circle):
        return f"WI {shape.radius_nm}NM RADIUS OF {coordinate(shape.lat, shape.lon)}."
    corners = [*shape.points, shape.points[0]]
    return "WI AREA BOUNDED BY " + " - ".join(coordinate(*corner) for corner in corners) + "."


def _reach_nm(parsed: ParsedNotam, shape: Circle | Polygon) -> float:
    """How far from the NOTAM's Q-circle centre the shape reaches, in NM."""
    if isinstance(shape, Circle):
        return distance_nm(parsed.centre_lat, parsed.centre_lon, shape.lat, shape.lon) + shape.radius_nm
    return max(distance_nm(parsed.centre_lat, parsed.centre_lon, lat, lon) for lat, lon in shape.points)


def drawn(parsed: ParsedNotam, read: NotamReading | None) -> ParsedNotam:
    """The NOTAM with the shape a hand-written reading used to carry written into its E text, where code now reads it (G16).

    A reading's `geometry` no longer locates a NOTAM; item E does. A reading with no geometry (or an empty polygon, which
    cannot be written) leaves the NOTAM as it was. A NOTAM whose E text already holds coordinates is left as it is: item E
    already draws it (a second drawing would be Broken). A real Q-circle always contains the area, and code treats a shape
    that reaches more than the slack outside it as a parser miss, so the Q-circle is widened (centre kept) to contain what is drawn.
    """
    shape = read.geometry if read else None
    if shape is None or (isinstance(shape, Polygon) and not shape.points) or not isinstance(area_from_item_e(parsed), NoArea):
        return parsed
    radius = max(parsed.radius_nm, _reach_nm(parsed, shape))
    return parsed.model_copy(update={"item_e": f"{parsed.item_e} {drawing(shape)}", "radius_nm": radius})


def as_written(shape: Circle) -> Circle:
    """The circle as code reads it back from the E text (coordinates to six decimals of a second), for an exact-edge test."""
    parsed = area_from_item_e(notam(circle=shape, item_e=drawing(shape)))
    assert isinstance(parsed, Shape)
    assert isinstance(parsed.area, Circle)
    return parsed.area


def act_on(parsed: ParsedNotam, read: NotamReading | None, lz: LandingZone, seg: ActionSegment, **kw):
    """notam_action with the reading's shape written into item E first (see `drawn`)."""
    return notam_action(drawn(parsed, read), read, lz, seg, **kw)


def reading(
    *,
    notam_id: str = "H7171/26",
    category: Category = Category.COOPERATIVE,
    activity: str = "drone ops",
    geometry: Circle | Polygon | None = None,
    floor: Limit = SFC,
    ceiling: Limit | None = None,
    schedule: str = "",
    contact: ContactKind = ContactKind.PHONE,
    value: str | None = "07784 546062",
    applies_to: AppliesTo = AppliesTo.ALL_AIRCRAFT,
    exemptions: list[Exemption] | None = None,
    designators: list[str] | None = None,
    reason: ProhibitiveReason = ProhibitiveReason.NONE,
    quote: str | None = None,
) -> NotamReading:
    return NotamReading(
        id=notam_id,
        category=category,
        activity=activity,
        geometry=geometry,
        floor=floor,
        ceiling=ceiling or amsl(800),
        schedule=schedule,
        applies_to=applies_to,
        exemptions=exemptions or [],
        contact=Contact(kind=contact, value=value),
        activates_designators=designators or [],
        prohibitive_reason=reason,
        prohibitive_quote=quote,
    )


def exemption(quote: str, covers: ExemptionCovers = ExemptionCovers.PARAMOTOR_CLASS) -> Exemption:
    return Exemption(quote=quote, covers=covers)


# --- assess(): sources and a calm default run ---------------------------------------------


NOW = utc(7, 30)  # the conftest clock: 2026-10-31T07:30Z, a Saturday; London is on GMT (the clocks went back on 25 Oct)
ISSUED = utc(7, 0)  # a PIB issued half an hour before NOW
ONE_HOUR = timedelta(hours=1)


def status(source: SourceName, state: SourceState = SourceState.OK, reason: str = "", issued_at: datetime | None = None) -> SourceStatus:
    """A hand-built SourceStatus. Failed and stale ones get a reason, as the clients give them."""
    if state is not SourceState.OK and not reason:
        reason = f"{source.value} {state.value} in this test"
    return SourceStatus(source=source, state=state, reason=reason, fetched_at=NOW, issued_at=issued_at)


def weather_forecast(
    first: datetime,
    last: datetime,
    *,
    changes: Mapping[datetime, Mapping[str, float | None]] | None = None,
    after: Mapping[datetime, Mapping[str, float | None]] | None = None,
    elevation_m: float | None = 9.0,
    state: SourceState = SourceState.OK,
) -> Forecast:
    """A calm forecast with a sample at every whole hour from first to last, both included.

    changes maps a timestamp to values that replace the calm ones (None removes one).
    after maps a timestamp to values applied to that timestamp and every later one.
    A failed state gives a Forecast with no samples, as the client does.
    """
    if state is not SourceState.OK:
        return Forecast(status=status(SourceName.WEATHER, state))
    samples = {}
    t = first
    while t <= last:
        values = {}
        for since, over in (after or {}).items():
            if t >= since:
                values.update(over)
        values.update((changes or {}).get(t, {}))
        samples[t] = calm(**values)
        t += ONE_HOUR
    return Forecast(status=status(SourceName.WEATHER), elevation_m=elevation_m, samples=samples)


def forecast_failed(state: SourceState = SourceState.FAILED) -> Forecast:
    """What the weather client returns when it could not get a forecast."""
    return Forecast(status=status(SourceName.WEATHER, state))


def airspace_fetch(*areas: Airspace, state: SourceState = SourceState.OK) -> AirspaceFetch:
    return AirspaceFetch(areas=[] if state is not SourceState.OK else list(areas), status=status(SourceName.AIRSPACE, state))


def pib_result(*notams: ParsedNotam, state: SourceState = SourceState.OK, issued: datetime = ISSUED) -> PibResult:
    """A PIB result. Failed has no NOTAMs. Issued sets the PIB's Issued time, which the stale rule reads."""
    return PibResult(status=status(SourceName.NOTAMS, state, issued_at=issued), notams=[] if state is SourceState.FAILED else list(notams))


def run_assess(
    *,
    window: tuple[datetime, datetime] | None = (utc(9), utc(12)),
    now: datetime = NOW,
    forecast: Forecast | None = None,
    airspace: AirspaceFetch | None = None,
    pib: PibResult | None = None,
    readings: Mapping[str, NotamReading] | None = None,
    reader_status: Mapping[str, str] | None = None,
    lat: float = LZ.lat,
    lon: float = LZ.lon,
    live: bool = False,
    stale_rule: bool | None = None,
    tz: str | None = None,
) -> Assessment:
    """assess() at Blue Mill on a calm day. By default: 09:00-12:00Z, calm weather, no airspace, no NOTAMs, all sources ok.

    window=None asks for the default window; the calm forecast then covers it.
    """
    if forecast is None:
        first, last = window or default_window(now)
        forecast = weather_forecast(first - ONE_HOUR, last + ONE_HOUR)
    return assess(
        lat,
        lon,
        now=now,
        forecast=forecast,
        airspace=airspace or airspace_fetch(),
        pib=pib or pib_result(),
        readings=readings,
        reader_status=reader_status,
        window=window,
        live=live,
        stale_rule=stale_rule,
        tz=tz,
    )


def wind(mph: float) -> dict[str, float]:
    """Calm weather with the 10 m wind at mph and a gust 2 mph above it (spread 2), and 80 m / 120 m wind kept near it."""
    return {"wind_10m": mph, "gust_10m": mph + 2, "wind_80m": mph + 1, "wind_120m": mph + 2}


# --- guidance snapshots ---------------------------------------------------------------------------------------------------


class EmbedderSpy:
    """An embedder factory (what `load_index(embedder_factory=...)` takes) that counts the times it is called and hands out the same embedder every time."""

    def __init__(self, embedder: object | None = None, error: Exception | None = None) -> None:
        self.embedder = FakeEmbedder() if embedder is None else embedder
        self.error = error
        self.constructed = 0

    def __call__(self) -> object:
        self.constructed += 1
        if self.error is not None:
            raise self.error
        return self.embedder


def guidance_chunk(doc: str, section: str, text: str, *, title: str = "", edition: str = "1") -> Chunk:
    """One chunk. The title defaults to one derived from the section so every chunk has one."""
    return Chunk(doc, edition, section, title or f"Part {section}", text, hashlib.sha256(text.encode("utf-8")).hexdigest())


def write_snapshot(
    folder: Path,
    chunks: Sequence[Chunk],
    *,
    embedder: object | None = None,
    model: str = SETTINGS.embedding_model,
    chunker_version: int = SETTINGS.chunker_version,
    stored_key: str | None = None,
    link_only: Sequence[str] = (),
    register: bool = True,
) -> str:
    """Write `manifest.yaml` and `embeddings.sqlite` (and `sources.yaml` when `register`) into `folder` with PR-7's writers; return the corpus key computed from the manifest.

    `stored_key` is what the embeddings file says its key is (default: the computed one). The model and chunker version default to settings.yaml's, so a snapshot's key matches what `load_index` computes. `link_only` are document ids the register marks `indexed: false`;
    they get no chunks of their own here unless `chunks` has some (a test of the read-side filter puts them there on purpose).
    """
    folder.mkdir(parents=True, exist_ok=True)
    embedder = FakeEmbedder() if embedder is None else embedder
    docs = sorted({c.doc for c in chunks})
    editions = {c.doc: c.edition for c in chunks}
    documents = [{"id": d, "edition": editions[d], "sha256": hashlib.sha256(d.encode()).hexdigest()} for d in docs]
    manifest = {"corpus_key": "", "model": model, "chunker_version": chunker_version, "documents": documents}
    key = corpus.corpus_key(manifest, model, chunker_version)
    manifest["corpus_key"] = key
    corpus.write_manifest(folder / corpus.MANIFEST_NAME, manifest)
    vectors = [list(v) for v in embedder.embed([c.text for c in chunks])] if chunks else []  # type: ignore[attr-defined]
    corpus.write_embeddings(folder / corpus.EMBEDDINGS_NAME, chunks, vectors, {"corpus_key": key if stored_key is None else stored_key, "model": model, "chunker_version": str(chunker_version)})
    if register:
        entries = [
            {"id": d, "title": d.title(), "url": f"https://docs.test/{d}.pdf", "edition": editions.get(d, "1"), "licence": "test", "review_date": "2026-11-07", "indexed": d not in link_only}
            for d in sorted({*docs, *link_only})
        ]
        (folder / "sources.yaml").write_text(yaml.safe_dump({"documents": entries}, sort_keys=False), encoding="utf-8")
    return key


def write_questions(path: Path, *questions: tuple[str, Sequence[tuple[str, str]]]) -> Path:
    """Write labelled questions: (question, [(document, section), ...])."""
    entries = [{"question": q, "expected": [{"document": d, "section": s} for d, s in expected]} for q, expected in questions]
    path.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
    return path


# --- the fake v1 dev experiment of the judge calibration tests (5a PR-8.F1) ---------------------------------------------------------


class FakeExperiment:
    """A v1 dev experiment as LangSmith would hold it: the project and one root run per dev example and rep, as the runner's `create_run` logs them."""

    def __init__(self, name: str, project: ls_schemas.TracerSessionResult, runs: list[ls_schemas.Run], records: list[dict]):
        self.name, self.project, self.runs, self.records = name, project, runs, records
        self.by_slot = {(meta(run)["example_id"], meta(run)["rep"]): run for run in runs}

    def run(self, example_id: str, rep: int) -> ls_schemas.Run:
        return self.by_slot[example_id, rep]

    def record(self, example_id: str) -> dict:
        return next(record for record in self.records if record["id"] == example_id)

    def path_of(self, run: ls_schemas.Run) -> str:
        return self.record(meta(run)["example_id"])["reference"]["path"]


def fake_experiment(
    name: str = "v1-dev-20261108-160700",
    *,
    reps: Sequence[int] = (1, 2, 3),
    errored: Sequence[tuple[str, int]] = (),
    missing: Sequence[tuple[str, int]] = (),
    splits: Mapping[tuple[str, int], str] | None = None,
    reference_dataset_id: uuid.UUID | None = None,
    project_metadata: Mapping[str, object] | None = None,
) -> FakeExperiment:
    """The committed dev examples, each run in every rep of `reps`. An invented answer per run. `errored` slots carry an error and no outputs, `missing` slots have no run."""
    records = load_dataset("dev").records
    splits = splits or {}
    started = datetime(2026, 11, 8, 16, 7, tzinfo=UTC)
    runs = []
    for rep in reps:
        for record in records:
            slot = (record["id"], rep)
            if slot in missing:
                continue
            failed = slot in errored
            runs.append(
                ls_schemas.Run(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{record['id']}/{rep}"),
                    name=f"v1-{record['id']}",
                    run_type="chain",
                    start_time=started,
                    end_time=started + timedelta(seconds=12 + rep),  # the run's own end time: an update that does not carry it moves the run in time
                    trace_id=uuid.uuid5(uuid.NAMESPACE_URL, f"trace/{name}/{record['id']}/{rep}"),
                    reference_example_id=example_uuid(record["id"]),
                    inputs={"question": record["inputs"]["question"]},
                    outputs=None if failed else {"answer": f"Invented answer to {record['id']} on rep {rep}.", "guard": "passed", "usage": None},
                    error="the graph returned no final answer" if failed else None,
                    tags=[f"rep:{rep}", "v1"],
                    extra={"metadata": {"rep": rep, "example_id": record["id"], "experiment": name, "split": splits.get(slot, "dev"), "attempts": 1}},
                )
            )
    project = ls_schemas.TracerSessionResult(
        id=uuid.uuid5(uuid.NAMESPACE_URL, f"project/{name}"),
        name=name,
        start_time=started,
        tenant_id=uuid.uuid5(uuid.NAMESPACE_URL, "tenant"),
        reference_dataset_id=reference_dataset_id,
        extra={"metadata": dict(project_metadata or {"agent": "v1", "split": "dev", "reps": len(reps)})},
    )
    return FakeExperiment(name, project, runs, records)


def meta(run: ls_schemas.Run) -> dict:
    """The runner's metadata on a fake run (rep, example_id, split, ...), for a test to read or change."""
    return (run.extra or {})["metadata"]


def _feedback(run: ls_schemas.Run, key: str, *, score: object, value: object, source: str, minute: int, comment: str | None = None) -> ls_schemas.Feedback:
    moment = datetime(2026, 11, 8, 17, minute, tzinfo=UTC)
    return ls_schemas.Feedback(
        id=uuid.uuid5(uuid.NAMESPACE_URL, f"feedback/{run.id}/{key}/{source}/{minute}"),
        run_id=run.id,
        trace_id=None,
        key=key,
        score=score,
        value=value,
        comment=comment,
        created_at=moment,
        modified_at=moment,
        feedback_source=ls_schemas.FeedbackSourceBase(type=source),
    )


def judge_feedback(run: ls_schemas.Run, criterion: str, passed: bool | None, *, minute: int = 0, comment: str | None = "JUDGE-REASON") -> ls_schemas.Feedback:
    """The judge's run-1 feedback as send_feedback logs it: `judge.<criterion>`, score 1.0 or 0.0 (None for a judge error), the judge's reason as the comment, source model."""
    return _feedback(run, f"judge.{criterion}", score=None if passed is None else float(passed), value=None, source="model", minute=minute, comment=comment)


def dan_feedback(run: ls_schemas.Run, key: str, value: object, *, score: object = None, source: str = "app", minute: int = 1) -> ls_schemas.Feedback:
    """A label as Dan submits it in the annotation queue: the bare criterion name, a pass/fail value, source app."""
    return _feedback(run, key, score=score, value=value, source=source, minute=minute)
