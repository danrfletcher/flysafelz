# FlySafeLZ

Is it safe and legal to **fly** from this landing zone, in this window? A map-and-chat agent for UK paramotor pilots: click any coordinates, and it checks the forecast (Open-Meteo), permanent airspace (openAIP) and NOTAMs, then returns one of three verdicts: **NO-GO**, **UNVERIFIED** (a source failed or a NOTAM could not be read, and nothing working rules the hour out) or **PILOT DISCRETION**. PILOT DISCRETION carries qualifiers: **APPROACHING LIMITS** (any caution), **WITH CONDITIONS** (conditions the pilot must meet), or both. It also returns the flyable window and a brief with the reasoning shown. The system never says "go" on its own: the pilot decides.

Decision support only, not a primary means of navigation. Whether you may *take off* from a given field (landowner permission, built-up areas, local hazards) is the pilot's job, not this agent's.

Status: being set up. The build steps are tracked in the FlySafeLZ Roadmap.

## Planned stack

Python 3.12 with `uv` · LangGraph + LangChain v1 · LangSmith (traces, datasets, experiments) · Shapely · any OpenAI-compatible model endpoint from `.env` · React + Vite + Leaflet UI over the LangGraph server.

## Setup and tests

```
make setup   # uv sync; needs no API key
make test    # uv run pytest, with sockets disabled
make eval-reader   # the reader eval: the only target that calls the model, so it needs the READER_* settings
```

Copy `.env.example` to `.env` for the openAIP key when fixtures are captured later. `.env` is gitignored. The NOTAM reader's model endpoint is four more lines in the same file (or in the environment, which wins): `READER_PROVIDER`, `READER_BASE_URL`, `READER_API_KEY` and `READER_MODEL`, any OpenAI-compatible endpoint. All four are required and nothing falls back to a default; only `src/flysafelz/reader/config.py` reads them, and only when the reader is first used. `make test` never needs them.

`make eval-reader` reads every fully labelled NOTAM in `evals/reader/labels.yaml` three times with the real model (no cache), scores the runs against the labels, checks the live Blue Mill case against its `readings.yaml`, and writes `evals/reader/runs/<time>/report.md` and `report.json` (gitignored), printing their paths. Harmful downgrades lead the report; they never change the exit code. With a `READER_*` setting missing it stops and names it.

## Assessing a landing zone

```
uv run python -m flysafelz.cli assess --case blue_mill_2026-10-31   # offline, from fixtures/<case>/
uv run python -m flysafelz.cli assess --lat 53.802797 --lon -0.086644 # live; needs OPENAIP_API_KEY in the environment or .env
```

One row per UTC hour (a segment edge inside an hour prints as sub-rows), then source statuses, flyable windows and the overall line. `--start` and `--end` (ISO 8601 with a timezone, end exclusive) narrow the window; `--debug` also lists the NOTAMs that were dropped or ignored, with the reason. A live run has no NOTAM readings, so a NOTAM over the LZ that needs one reads UNVERIFIED. There is no flag that changes a limit. The committed outputs for both cases are in `tests/golden/`; `uv run pytest --update-golden` rewrites them.

Case mode and the age of the PIB. The archive the cases come from (Jonty/uk-notam-archive) snapshots the PIB only about twice a day, at roughly 02:20 and 09:30 UTC, so a case's captured bulletin can be about 5 h old at the case's `now`. Case mode therefore treats the captured PIB as current at `now`: the 3 h `pib_stale_hours` rule is not applied, the NOTAMs source reads `ok`, and its line shows the PIB's issued time with a "(case mode: captured PIB treated as current at the case's now)" note so the real age is visible. Live mode keeps the 3 h rule unchanged. A case can opt back in with `stale_rule: live` in its `case.yaml`; only `fixtures/stale_pib_constructed/` does, so the stale, UNVERIFIED path stays tested end to end. Any other value is a load error.

## Rules

Every threshold lives in `src/flysafelz/rules.yaml` with its source and confidence. Python code holds no limits of its own. This table is the output of `uv run python -m flysafelz.cli rules`, and a test keeps the two identical.

| id | value | unit | comparison | source | confidence | note |
| --- | --- | --- | --- | --- | --- | --- |
| wind_10m_caution | 12 | mph | >= | FlySafeLZ Spec s2, Plan 2 G22 (Dan, Q3); typical paramotor school wind limits | best practice | Mean wind at 10 m. Caution band below the 10 m NO_GO limit. |
| wind_10m_no_go | 16 | mph | > | FlySafeLZ Spec s2, Plan 2 G22 (Dan, Q3); typical paramotor school wind limits | best practice | Mean wind at 10 m. 16.0 mph is APPROACHING LIMITS; NO_GO needs more than 16.0 mph. |
| gust_spread_caution | 4 | mph | >= | FlySafeLZ Spec s2, Plan 2 G22 (Dan, Q3) | educated guess | Gust minus mean wind at 10 m, worst of the row's timestamps. |
| gust_spread_no_go | 6 | mph | > | FlySafeLZ Spec s2, Plan 2 G22 (Dan, Q3) | educated guess | Gust minus mean wind at 10 m. Exactly 6.0 mph is not NO_GO; the comparison is strict. |
| wind_aloft_caution | 16 | mph | >= | FlySafeLZ Spec s2, Plan 2 G2/G22 (Dan, Q3): same number as the 10 m NO_GO limit | educated guess | Wind at 80 m or 120 m. Caution only; the hard limits are at 10 m. |
| gradient_caution | 9 | mph | >= | FlySafeLZ Spec s2, Plan 2 G2/G22 (Dan, Q3) | educated guess | Absolute difference between 120 m and 10 m wind. |
| gradient_no_go | 16 | mph | > | FlySafeLZ Spec s2, Plan 2 G2/G22 (Dan, Q3) | educated guess | Absolute difference between 120 m and 10 m wind. Exactly 16.0 mph is APPROACHING LIMITS; NO_GO needs more than 16.0 mph; the comparison is strict. |
| precipitation_no_go | 0.1 | mm/h | > | FlySafeLZ Spec s2, Plan 2 G22 | educated guess | Rain in the preceding hour, read at the hour's closing timestamp. 0.0 mm/h is GO; 0.1 mm/h (trace forecast rain) is APPROACHING LIMITS; 0.2 mm/h or more (sustained rain) is NO_GO because of wet-wing deep-stall risk and prop erosion. |
| rain_chance_caution | 50 | % | >= | FlySafeLZ Spec s2, Plan 2 G22 | educated guess | Precipitation probability. Optional variable: skipped and noted when missing. |
| visibility_caution | 5000 | m | <= | Standard 5 km VFR visibility figure (UK SERA.5001 is the reference to check); Plan 2 G22 | best practice | Not checked against the regulation text. Exactly 5000 m is APPROACHING LIMITS; the comparison includes the limit. |
| visibility_no_go | 1500 | m | < | Plan 2 G22; compare the Class G flight-visibility minimum for slow aircraft in UK SERA.5001 | best practice | Not checked against the regulation text; confirm before relying on it as sourced. |
| low_cloud_caution | 90 | % | >= | FlySafeLZ Spec s2, Plan 2 G22 | educated guess | Low cloud cover (Open-Meteo cloud_cover_low). |
| thermic_shortwave_caution | 600 | W/m² | >= | FlySafeLZ Spec s2, Plan 2 G3/G22 (Dan, Q2) | educated guess | Thermic caution needs this AND the thermic CAPE limit together. |
| thermic_cape_caution | 300 | J/kg | >= | FlySafeLZ Spec s2, Plan 2 G3/G22 (Dan, Q2) | educated guess | Thermic caution needs this AND the shortwave limit together. |
| cape_no_go | 1000 | J/kg | > | Common convention that CAPE of about 1,000 J/kg or more is meaningful convective instability; Plan 2 G22 (Dan, Q2) | best practice | Convective instability and thunderstorm risk. Exactly 1000 J/kg is APPROACHING LIMITS; NO_GO needs more than 1000 J/kg; the comparison is strict. |
| ceiling_left_no_go | 300 | ft | < | FlySafeLZ Spec s2, Plan 2 E15/G22 | educated guess | Airspace ceiling (floor QNH minus the buffer) minus site elevation. Under this, no room to fly. |
| ceiling_buffer | 200 | ft | = | FlySafeLZ Spec s2, Plan 2 primitives | educated guess | Margin kept below an airspace floor when stating 'stay below X ft QNH'. |
| fl_ft_per_hpa | 27 | ft/hPa | = | ICAO standard atmosphere: about 27 ft per hPa near sea level | sourced | Converts a flight level to feet against the forecast pressure_msl. |
| fl_reference_hpa | 1013.25 | hPa | = | ICAO standard atmosphere sea-level pressure | sourced | Flight levels are measured from this pressure. |
| local_area | 2 | NM | = | FlySafeLZ Spec s2, Plan 2 primitives (Dan) | educated guess | Radius of the local flying area around the landing zone. Not user-adjustable. |
| airspace_box_nm | 10 | NM | = | FlySafeLZ Spec s2, Plan 2 G7 (wider than the 2 NM local area so the map has context) | educated guess | Half-width of the box around the LZ that the airspaces are cut down to. openAIP is asked for all of country=GB (its firewall refuses a bbox) and every airspace touching the box is kept. |
| band_low | 5000 | ft | = | FlySafeLZ Spec s2, Plan 2 primitives (Dan) | educated guess | Airspace floors below this give a ceiling condition; above it, a warning. |
| band_high | 10000 | ft | = | FlySafeLZ Spec s2, Plan 2 primitives (Dan) | educated guess | Airspace floors above this are awareness only (map and collapsed list). |
| verdict_horizon_day | 3 | day | = | FlySafeLZ Spec s2, Plan 2 Q9 (Dan); day 0 is today in Europe/London | educated guess | Last day that can carry a verdict, if every required weather variable is present. |
| outlook_horizon_day | 7 | day | = | Vector check of Open-Meteo ukmo_seamless on 2026-11-06: 10 m wind, gust, CAPE and visibility run to day 7 | sourced | Beyond this day nothing is fetched and the hour is TOO_FAR. |
| pib_stale_hours | 3 | h | = | FlySafeLZ Spec s2, Plan 2 E2 | educated guess | Live mode: a PIB issued longer ago than this makes NOTAMs stale, so UNVERIFIED. Case mode treats the captured PIB as current at the case's now, because the archive snapshots it only about twice a day. |
| uk_bounds_lat_min | 49.8 | deg | < | Plan 2 E16 (south of the Isles of Scilly) | educated guess | A latitude below this is outside UK coverage. Exactly this latitude is inside. |
| uk_bounds_lat_max | 61 | deg | > | Plan 2 E16 (north of Shetland) | educated guess | A latitude above this is outside UK coverage. Exactly this latitude is inside. |
| uk_bounds_lon_min | -8.7 | deg | < | Plan 2 E16 (west of Northern Ireland) | educated guess | A longitude (east positive) below this is outside UK coverage. Exactly this longitude is inside. |
| uk_bounds_lon_max | 2 | deg | > | Plan 2 E16 (east of East Anglia) | educated guess | A longitude (east positive) above this is outside UK coverage. Exactly this longitude is inside. |
| civil_twilight_depression | 6 | deg | = | Standard definition of civil twilight: sun centre 6 degrees below the horizon; Plan 2 G6 | sourced | Angle below the horizon for civil dawn and civil dusk. Outside civil twilight is night. |
| notam_max_lower_fl | 195 | FL | > | FlySafeLZ Spec s2, Plan 2 G13; Class G in the UK runs up to FL195 | educated guess | A NOTAM whose Q-line lower limit is above this is dropped by the prefilter. |
| notam_ignore_subject_letters | L, F, M, N, I, C, S, O | subject first letter | in | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | best practice | Q-line subjects whose first letter cannot affect a paramotor (lighting, aerodrome facilities, movement area, nav aids, comms, ATS services, other). A, R and W are never in this list. |
| notam_keep_subjects | FA, OB, OL | subject | in | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | best practice | Subjects kept although their first letter is in the ignore list: aerodrome warnings (drones, fireworks, lasers, live ordnance) and obstacles (cranes, rigs). |
| notam_wide_ignore_subjects | AF | subject | in | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | educated guess | FIR information subjects. At or above the area-wide radius they are FIR-wide AIS admin (e.g. MIL AIP checklists) and are ignored; below it (transponder mandatory zones) they are kept. |
| notam_ignore_traffic | I | Q-line traffic | in | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | best practice | Traffic exactly I (IFR only) cannot affect a paramotor. IV and V are kept. |
| notam_ignore_purpose | K | Q-line purpose | in | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | sourced | Purpose K marks a checklist or status NOTAM. |
| notam_ignore_security_subjects | RO | subject | in | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | best practice | In series V only: security advice to air operators. Series V with any other subject (e.g. volcanic ash) is kept. |
| notam_restriction_subjects | RD, RR, RP, RT | subject | in | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | best practice | Danger, restricted, prohibited and temporary reserved airspace. With a designator in item E the NOTAM is handled through the airspace polygon; without one it goes to the reader. |
| notam_awareness_subjects | GW | subject | in | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | best practice | GNSS or radio-frequency jamming: awareness only, no reader. |
| notam_area_wide_nm | 50 | NM | >= | ICAO Doc 8126 / UK AIP GEN 3.1 §3.6.3.4 | educated guess | A Q-line radius at or above this is an area-wide NOTAM: awareness only (and FIR-wide AIS admin for the subjects in notam_wide_ignore_subjects). A radius of 999 means the whole FIR. |
| capture_trim_nm | 25 | NM | > | FlySafeLZ Spec s2, Plan 2 G20 (fixture PIBs are trimmed to 25 NM) | educated guess | make fixtures keeps a NOTAM only when its Q-circle edge is within this distance of the LZ, so a fixture stays small but covers well past the 2 NM local area. A NOTAM whose Q-circle cannot be read is kept. |
| reader_area_slack_nm | 1 | NM | > | Ticket 3 Q4 (Dan); 4 archived PIBs (4 Jul, 22 Aug, 19 Sep, 31 Oct 2026): the area in the E text is never more than 1 NM outside the Q-circle | educated guess | A NOTAM area built from the E text that reaches further than this outside the Q-circle is treated as a parser miss and replaced by the Q-circle. Exactly this far out is kept. |

Confidence is `sourced` (a citable document), `best practice` (common practice, not one citation) or `educated guess` (our own call). Limits cannot be changed by the user.

## Limitations

- Airspace comes from openAIP only. It is not cross-checked against the official UK AIP, so a difference between the two will not be caught.
- A danger area whose openAIP record has no activation hours is treated as active (over-cautious on purpose).
