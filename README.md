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
```

Copy `.env.example` to `.env` for the openAIP key when fixtures are captured later. `.env` is gitignored.

## Rules

Every threshold lives in `src/flysafelz/rules.yaml` with its source and confidence. Python code holds no limits of its own. This table is the output of `uv run python -m flysafelz.cli rules`, and a test keeps the two identical.

| id | value | unit | comparison | source | confidence | note |
| --- | --- | --- | --- | --- | --- | --- |
| wind_10m_caution | 12 | mph | >= | FlySafeLZ Spec s2, Plan 2 G22 (Dan, Q3); typical paramotor school wind limits | best practice | Mean wind at 10 m. Caution band below the 10 m NO_GO limit. |
| wind_10m_no_go | 16 | mph | >= | FlySafeLZ Spec s2, Plan 2 G22 (Dan, Q3); typical paramotor school wind limits | best practice | Mean wind at 10 m. 16.0 mph is NO_GO. |
| gust_spread_caution | 4 | mph | >= | FlySafeLZ Spec s2, Plan 2 G22 (Dan, Q3) | educated guess | Gust minus mean wind at 10 m, worst of the row's timestamps. |
| gust_spread_no_go | 6 | mph | > | FlySafeLZ Spec s2, Plan 2 G22 (Dan, Q3) | educated guess | Gust minus mean wind at 10 m. Exactly 6.0 mph is not NO_GO; the comparison is strict. |
| wind_aloft_caution | 16 | mph | >= | FlySafeLZ Spec s2, Plan 2 G2/G22 (Dan, Q3): same number as the 10 m NO_GO limit | educated guess | Wind at 80 m or 120 m. Caution only; the hard limits are at 10 m. |
| gradient_caution | 9 | mph | >= | FlySafeLZ Spec s2, Plan 2 G2/G22 (Dan, Q3) | educated guess | Absolute difference between 120 m and 10 m wind. |
| gradient_no_go | 16 | mph | >= | FlySafeLZ Spec s2, Plan 2 G2/G22 (Dan, Q3) | educated guess | Absolute difference between 120 m and 10 m wind. |
| precipitation_no_go | 0.1 | mm/h | >= | FlySafeLZ Spec s2, Plan 2 G22 | educated guess | Any measurable rain in the preceding hour. Applies to the hour's closing timestamp. |
| rain_chance_caution | 50 | % | >= | FlySafeLZ Spec s2, Plan 2 G22 | educated guess | Precipitation probability. Optional variable: skipped and noted when missing. |
| visibility_caution | 5000 | m | < | Standard 5 km VFR visibility figure (UK SERA.5001 is the reference to check); Plan 2 G22 | best practice | Not checked against the regulation text. |
| visibility_no_go | 1500 | m | < | Plan 2 G22; compare the Class G flight-visibility minimum for slow aircraft in UK SERA.5001 | best practice | Not checked against the regulation text; confirm before relying on it as sourced. |
| low_cloud_caution | 90 | % | >= | FlySafeLZ Spec s2, Plan 2 G22 | educated guess | Low cloud cover (Open-Meteo cloud_cover_low). |
| thermic_shortwave_caution | 600 | W/m² | >= | FlySafeLZ Spec s2, Plan 2 G3/G22 (Dan, Q2) | educated guess | Thermic caution needs this AND the thermic CAPE limit together. |
| thermic_cape_caution | 300 | J/kg | >= | FlySafeLZ Spec s2, Plan 2 G3/G22 (Dan, Q2) | educated guess | Thermic caution needs this AND the shortwave limit together. |
| cape_no_go | 1000 | J/kg | >= | Common convention that CAPE of about 1,000 J/kg or more is meaningful convective instability; Plan 2 G22 (Dan, Q2) | best practice | Convective instability and thunderstorm risk. |
| ceiling_left_no_go | 300 | ft | < | FlySafeLZ Spec s2, Plan 2 E15/G22 | educated guess | Airspace ceiling (floor QNH minus the buffer) minus site elevation. Under this, no room to fly. |
| ceiling_buffer | 200 | ft | = | FlySafeLZ Spec s2, Plan 2 primitives | educated guess | Margin kept below an airspace floor when stating 'stay below X ft QNH'. |
| fl_ft_per_hpa | 27 | ft/hPa | = | ICAO standard atmosphere: about 27 ft per hPa near sea level | sourced | Converts a flight level to feet against the forecast pressure_msl. |
| fl_reference_hpa | 1013.25 | hPa | = | ICAO standard atmosphere sea-level pressure | sourced | Flight levels are measured from this pressure. |
| local_area | 2 | NM | = | FlySafeLZ Spec s2, Plan 2 primitives (Dan) | educated guess | Radius of the local flying area around the landing zone. Not user-adjustable. |
| band_low | 5000 | ft | = | FlySafeLZ Spec s2, Plan 2 primitives (Dan) | educated guess | Airspace floors below this give a ceiling condition; above it, a warning. |
| band_high | 10000 | ft | = | FlySafeLZ Spec s2, Plan 2 primitives (Dan) | educated guess | Airspace floors above this are awareness only (map and collapsed list). |
| verdict_horizon_day | 3 | day | = | FlySafeLZ Spec s2, Plan 2 Q9 (Dan); day 0 is today in Europe/London | educated guess | Last day that can carry a verdict, if every required weather variable is present. |
| outlook_horizon_day | 7 | day | = | Vector check of Open-Meteo ukmo_seamless on 2026-11-06: 10 m wind, gust, CAPE and visibility run to day 7 | sourced | Beyond this day nothing is fetched and the hour is TOO_FAR. |
| pib_stale_hours | 3 | h | = | FlySafeLZ Spec s2, Plan 2 E2 | educated guess | A PIB issued longer ago than this makes NOTAMs stale, so UNVERIFIED. |

Confidence is `sourced` (a citable document), `best practice` (common practice, not one citation) or `educated guess` (our own call). Limits cannot be changed by the user.
