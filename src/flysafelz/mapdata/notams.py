"""The live PIB's NOTAMs active from now to 24 hours ahead as GeoJSON features, for GET /map/notams.

    notam_features(client, now)   -> the features, or raises RefreshFailed
    in_window(notam, now)         -> True when the NOTAM is active at some moment of [now, now + 24 h)

The PIB comes from `fetch_pib("live", now)` and is read with `pib.parse_pib`, which drops what a later R or C replaced or cancelled. A NOTAM is
kept when it starts before now + 24 h and ends after now, or has no end (permanent). The window is half open: a NOTAM that starts at exactly
now + 24 h is out, and so is one that ends at exactly now.

Every NOTAM in the window is served at every height, anywhere in the UK. The agent's per-landing-zone prefilter (`notam_max_lower_fl` and the
local area) is not applied: the map's "all heights" layers and the "Above FL100" group need the high ones.

A feature's geometry is `shapes.notam_geometry`, the one shape a NOTAM has everywhere; the route never builds a shape. `approx` is true exactly
when that function fell back to the NOTAM's Q-circle. The properties: id, q_code, lower_fl, upper_fl, item_f and item_g (the F and G limit
text, null when absent), valid_from and valid_to (ISO UTC, valid_to null when permanent), permanent, est, item_d (schedule text, null when
absent), text (item E exactly as published, real phone numbers included: public data, needed to call first; it is never logged) and approx.
"""

from datetime import datetime, timedelta

import httpx

from flysafelz.clients.pib_source import fetch_pib, utc_text
from flysafelz.contracts import ParsedNotam, SourceState
from flysafelz.mapdata.cache import RefreshFailed
from flysafelz.pib import PibError, parse_pib
from flysafelz.shapes import notam_geometry

WINDOW = timedelta(hours=24)
UNREADABLE = "NATS PIB: unreadable"  # a fixed line: the parser's own message can quote the PIB


def in_window(notam: ParsedNotam, now: datetime) -> bool:
    ends_after_now = notam.valid_to is None or notam.valid_to > now
    return notam.valid_from < now + WINDOW and ends_after_now


def notam_feature(notam: ParsedNotam) -> dict:
    geometry, approx = notam_geometry(notam)
    properties = {
        "id": notam.id,
        "q_code": notam.q_code,
        "lower_fl": notam.lower_fl,
        "upper_fl": notam.upper_fl,
        "item_f": notam.item_f,
        "item_g": notam.item_g,
        "valid_from": utc_text(notam.valid_from),
        "valid_to": None if notam.valid_to is None else utc_text(notam.valid_to),
        "permanent": notam.permanent,
        "est": notam.est,
        "item_d": notam.item_d,
        "text": notam.item_e,
        "approx": approx,
    }
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def notam_features(client: httpx.Client, now: datetime) -> list[dict]:
    fetched = fetch_pib("live", now, client=client)
    if fetched.status.state is not SourceState.OK or fetched.xml is None:
        raise RefreshFailed(fetched.status.reason or "NATS PIB: not fetched")
    try:
        _issued, notams = parse_pib(fetched.xml)
    except PibError as error:
        raise RefreshFailed(UNREADABLE) from error
    return [notam_feature(notam) for notam in notams if in_window(notam, now)]
