"""Builders for the reader eval's tests: a NOTAM whose area is the Q-circle round Blue Mill, and a ReaderResult around a reading.

Nothing here is mocked: notam_action() and the geometry are the real ones. Ids are obviously fake (X0001/26) and the one phone
number is in the Ofcom drama range (07700 900xxx), so a leak into a finding is easy to see.
"""

from factories import circle_at, notam

from flysafelz.contracts import Circle, NotamReading, ParsedNotam
from flysafelz.reader.contract import ReaderResult, ReaderStatus

FAKE_PHONE = "07700 900123"
PLACEHOLDER = "<PHONE_1>"
NO_SHAPE_E = "TEST AREA ACTIVE. NO SHAPE IS GIVEN IN THIS TEXT."  # build_area finds no area in it, so the Q-circle is the area


def q_notam(notam_id: str = "X0001/26", radius_nm: float = 3.0, *, circle: Circle | None = None, **overrides: object) -> ParsedNotam:
    """A NOTAM active all of 29 Oct to 10 Nov whose area is the Q-circle (`circle`, else radius_nm round Blue Mill): E gives no shape."""
    return notam(notam_id, circle=circle or circle_at(0, radius_nm=radius_nm), **{"item_e": NO_SHAPE_E, **overrides})


def read_result(reading: NotamReading | None, status: ReaderStatus | None = None, notam_id: str = "X0001/26") -> ReaderResult:
    """A ReaderResult for a reading (READ), or an unread one (FAILED) when reading is None."""
    if status is None:
        status = ReaderStatus.FAILED if reading is None else ReaderStatus.READ
    return ReaderResult(
        notam_id=notam_id,
        reading=reading,
        status=status,
        failed_checks=[] if reading is not None else ["the answer was wrong twice"],
        model="fake-model",
        prompt_version="test",
        latency_s=0.0,
        tokens=None,
    )
