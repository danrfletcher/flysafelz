# FlySafeLZ

Is it safe and legal to **fly** from this landing zone, in this window? A map-and-chat agent for UK paramotor pilots: click any coordinates, and it checks the forecast (Open-Meteo), permanent airspace (openAIP) and NOTAMs, then returns **NO-GO / GO WITH CONDITIONS / GO** (or **MARGINAL**, the pilot's call), the flyable window and a brief with the reasoning shown.

Decision support only, not a primary means of navigation. Whether you may *take off* from a given field (landowner permission, built-up areas, local hazards) is the pilot's job, not this agent's.

Status: being set up. The build steps are tracked in the FlySafeLZ Roadmap.

## Planned stack

Python 3.12 with `uv` · LangGraph + LangChain v1 · LangSmith (traces, datasets, experiments) · Shapely · any OpenAI-compatible model endpoint from `.env` · React + Vite + Leaflet UI over the LangGraph server.
