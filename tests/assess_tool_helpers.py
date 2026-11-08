"""What the assess tool tests share: an Assessment from ticket 2's real `assess()` over the factories' sources, written through the tool's own `build`."""

from datetime import datetime

from factories import NOW, airspace_fetch, pib_result, run_assess, utc, weather_forecast

from flysafelz.agent.tools.artifacts import AssessArtifact
from flysafelz.agent.tools.assess_tool import Inputs, build
from flysafelz.agent.window import Window
from flysafelz.contracts import Assessment, Verdict, display_verdict

LONDON = "Europe/London"
SIX = {
    display_verdict(Verdict.NO_GO, False, False),
    display_verdict(Verdict.UNVERIFIED, False, False),
    *{display_verdict(Verdict.PILOT_DISCRETION, limits, conditions) for limits in (False, True) for conditions in (False, True)},
}  # the six strings display_verdict gives


def window_of(start: datetime, end: datetime) -> Window:
    return Window(start, end, "test window", assumed=False)


def render(assessment: Assessment, window: tuple[datetime, datetime], *, tz: str = LONDON, now: datetime = NOW) -> tuple[str, AssessArtifact]:
    """The tool's text and artifact for an assessment, with no overlays and no phone book."""
    return build(assessment, Inputs(tz=tz, window=window_of(*window), now=now, overlays=[]))


def assess_calm(first: datetime, last: datetime, *, now: datetime = NOW, **kwargs) -> Assessment:
    """Ticket 2's assess() over the window [first, last) with calm weather and nothing else, as the tool calls it (explicit UTC window)."""
    return run_assess(window=(first, last), now=now, **kwargs)


__all__ = ["LONDON", "NOW", "SIX", "airspace_fetch", "assess_calm", "pib_result", "render", "utc", "weather_forecast", "window_of"]
